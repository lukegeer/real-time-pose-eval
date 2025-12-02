import torch
import torch.nn as nn


class PerJointDynamics(nn.Module):
    """
    Small per-joint GRU that outputs residuals (dv, da) and a gain logit, conditioned on global pose context.
    State is (x, y, vx, vy, ax, ay); we update vx/ax with learned residuals and integrate.
    """

    def __init__(self, input_size=6, hidden_size=32, global_size=6):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + global_size, 5)  # dvx, dvy, dax, day, gain_logit

    def forward(self, seq, global_feat):
        # seq: [B, T, 6]; global_feat: [B, global_size]
        _, h = self.gru(seq)  # h: [1, B, H]
        h = h[-1]
        h_cat = torch.cat([h, global_feat], dim=-1)
        out = self.fc(h_cat)  # [B,5]
        dvx, dvy, dax, day, gain_logit = torch.chunk(out, 5, dim=-1)
        gain = torch.sigmoid(gain_logit)
        return dvx.squeeze(-1), dvy.squeeze(-1), dax.squeeze(-1), day.squeeze(-1), gain.squeeze(-1)


class LearnedKalmanDynamics(nn.Module):
    """
    Learned predict + gain: for each joint, update velocity/accel with residuals and blend with measurement via a learned gain.
    """

    def __init__(self, num_joints=17, hidden_size=32, history=3, dt=1 / 30):
        super().__init__()
        self.num_joints = num_joints
        self.history = history
        self.dt = dt
        self.joint_nets = nn.ModuleList(
            [PerJointDynamics(input_size=6, hidden_size=hidden_size) for _ in range(num_joints)]
        )

    def forward(self, state_seq, meas=None):
        """
        Args:
            state_seq: [B, T, J, 6] of past states (T>=history)
            meas: optional measurement state [B, J, 6] (only x,y typically valid)
        Returns:
            pred: [B, J, 6] predicted/updated state
            gains: [B, J] learned gains used for blending
        """
        B, T, J, _ = state_seq.shape
        assert J == self.num_joints
        seq = state_seq[:, -self.history :, :, :]
        preds = []
        gains = []
        global_feat = seq[:, -1].mean(dim=1)  # [B,6] mean over joints at last step
        for j in range(J):
            dvx, dvy, dax, day, gain = self.joint_nets[j](seq[:, :, j, :], global_feat)
            prev = seq[:, -1, j, :]
            vx_new = prev[:, 2] + dvx
            vy_new = prev[:, 3] + dvy
            ax_new = prev[:, 4] + dax
            ay_new = prev[:, 5] + day
            x_new = prev[:, 0] + vx_new * self.dt + 0.5 * ax_new * (self.dt ** 2)
            y_new = prev[:, 1] + vy_new * self.dt + 0.5 * ay_new * (self.dt ** 2)

            if meas is not None:
                mx = meas[:, j, 0]
                my = meas[:, j, 1]
                x_new = gain * mx + (1 - gain) * x_new
                y_new = gain * my + (1 - gain) * y_new

            preds.append(torch.stack([x_new, y_new, vx_new, vy_new, ax_new, ay_new], dim=-1))
            gains.append(gain)
        pred = torch.stack(preds, dim=1)
        gains = torch.stack(gains, dim=1)
        return pred, gains


def rollout(model, init_state, gt_states, mask=None):
    """
    Autoregressive rollout using learned dynamics.
    init_state: [B, J, 6]
    gt_states: [B, T, J, 6] ground truth states for supervision
    mask: [B, T, J] optional visibility mask (1=valid)
    Returns preds: [B, T, J, 6]
    """
    B, T, J, _ = gt_states.shape
    preds = []
    history = [init_state]
    for t in range(T):
        hist_tensor = torch.stack(history, dim=1)  # [B, hist, J, 6]
        meas = gt_states[:, t] if mask is not None else None
        pred, _ = model(hist_tensor, meas=meas)
        preds.append(pred)
        history.append(pred)
        if len(history) > model.history:
            history.pop(0)
    preds = torch.stack(preds, dim=1)
    if mask is not None:
        preds = preds * mask.unsqueeze(-1) + gt_states * (1 - mask.unsqueeze(-1))
    return preds
