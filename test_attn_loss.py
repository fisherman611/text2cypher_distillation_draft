import torch
import sys

# Đảm bảo console in Tiếng Việt được trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def align_attention_heads(student_attn, teacher_attn, reduction="mean"):
    """
    Make student/teacher attention head dimensions compatible.
    Here we just average over heads.
    """
    if reduction == "mean" or student_attn.size(1) != teacher_attn.size(1):
        student_attn = student_attn.mean(dim=1, keepdim=True)
        teacher_attn = teacher_attn.mean(dim=1, keepdim=True)
    return student_attn, teacher_attn

def masked_attention_cka_loss(student_attn, teacher_attn, pair_mask, eps=1e-8):
    """
    CKA loss between masked attention maps.
    So sánh ma trận đặc trưng giữa student và teacher.
    CKA sẽ cho giá trị từ 0 đến 1 (1 là giống hệt nhau). Loss = 1 - CKA.
    
    Giải thích vì sao có thể bé/lớn:
    - Nếu CKA ~ 0.99, loss = 0.01 (rất bé).
    - Vì CKA chuẩn hóa scale nên nó không bị ảnh hưởng bởi việc giá trị attention bé. 
      Do đó, CKA loss phản ánh sự khác biệt về "cấu trúc" tốt hơn MSE.
    """
    s = (student_attn.float() * pair_mask).mean(dim=1)  # [B, L, L]
    t = (teacher_attn.float() * pair_mask).mean(dim=1)  # [B, L, L]

    row_mask = pair_mask.squeeze(1).bool()
    s_mass = s.abs().sum(dim=-1)
    t_mass = t.abs().sum(dim=-1)
    valid_rows = row_mask.any(dim=-1) & (s_mass > eps) & (t_mass > eps)
    if valid_rows.sum() < 3:
        return student_attn.new_tensor(0.0)

    s_rows = s[valid_rows]
    t_rows = t[valid_rows]
    col_mask = row_mask[valid_rows]
    valid_cols = col_mask.any(dim=0)
    if valid_cols.sum() < 2:
        return student_attn.new_tensor(0.0)

    x = s_rows[:, valid_cols]
    y = t_rows[:, valid_cols]
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    hsic = (x.T @ y).pow(2).sum()
    x_norm = (x.T @ x).pow(2).sum().sqrt()
    y_norm = (y.T @ y).pow(2).sum().sqrt()
    denom = x_norm * y_norm
    if denom <= eps:
        return x.new_tensor(0.0 if torch.allclose(x, y, atol=eps, rtol=0.0) else 1.0)

    cka = hsic / denom
    loss = 1.0 - cka.clamp(min=0.0, max=1.0)
    return loss

def masked_attention_distribution_loss(student_attn, teacher_attn, pair_mask, loss_type="raw_mse", eps=1e-8):
    """
    Hàm tổng hợp để tính các loại loss khác nhau.
    """
    student_attn, teacher_attn = align_attention_heads(student_attn, teacher_attn)
    pair_mask = pair_mask.to(student_attn.device)

    print(f"\n--- Tính toán với loss_type: {loss_type} ---")

    if loss_type == "raw_mse":
        """
        raw_mse: Trung bình bình phương khoảng cách (MSE) trên các phần tử được mask.
        Giải thích vì sao bé: 
        - Giá trị attention softmax thường nằm trong khoảng (0, 1), đặc biệt khi chuỗi dài, phần lớn các token có attention cực nhỏ (e.g., 1e-4).
        - Bình phương của (1e-4) là 1e-8.
        - Khi sum lên và chia cho tổng số pixel (mask.sum()), giá trị trung bình sẽ cực kỳ nhỏ (thường ~ 1e-6 hoặc 1e-5).
        """
        diff = ((student_attn.float() - teacher_attn.float()) ** 2) * pair_mask
        val_diff = diff.sum().item()
        val_denom = pair_mask.sum().clamp(min=1.0).item()
        loss = diff.sum() / pair_mask.sum().clamp(min=1.0)
        print(f"Tổng bình phương khác biệt (Tử số): {val_diff:.8e}")
        print(f"Tổng số ptử trong mask (Mẫu số): {val_denom}")
        print(f"Raw MSE Loss: {loss.item():.8e}")
        return loss

    if loss_type == "mass_mse":
        """
        mass_mse: MSE nhưng chia cho tổng khối lượng attention của teacher trong vùng mask.
        Giải thích:
        - Giúp loss lớn hơn một chút so với raw_mse nếu vùng mask có teacher attention không tập trung cao.
        - Tuy nhiên, tử số (diff^2) vẫn cực nhỏ do giá trị attention bé. 
        """
        diff = ((student_attn.float() - teacher_attn.float()) ** 2) * pair_mask
        denom = (teacher_attn.float() * pair_mask).sum().clamp(min=eps)
        val_diff = diff.sum().item()
        val_denom = denom.item()
        loss = diff.sum() / denom
        print(f"Tổng bình phương khác biệt (Tử số): {val_diff:.8e}")
        print(f"Tổng khối lượng attention teacher trong mask (Mẫu số): {val_denom:.8e}")
        print(f"Mass MSE Loss: {loss.item():.8e}")
        return loss

    if loss_type == "cka":
        loss = masked_attention_cka_loss(student_attn, teacher_attn, pair_mask, eps)
        print(f"CKA Loss: {loss.item():.8f} (Khoảng từ 0 đến 1, 0 là giống hệt)")
        return loss

    # Phần chuẩn hóa phân phối theo từng dòng
    s = student_attn.float() * pair_mask
    t = teacher_attn.float() * pair_mask
    s_sum = s.sum(dim=-1, keepdim=True)
    t_sum = t.sum(dim=-1, keepdim=True)
    valid_rows = (pair_mask.any(dim=-1, keepdim=True) & (s_sum > eps) & (t_sum > eps)).float()

    if valid_rows.sum() == 0:
        return student_attn.new_tensor(0.0)

    s_dist = s / s_sum.clamp(min=eps)
    t_dist = t / t_sum.clamp(min=eps)

    if loss_type == "mse":
        """
        mse (sau khi chuẩn hóa): 
        - Tại vùng mask, tổng các xác suất theo hàng được chuẩn hóa lại về 1.
        - Điều này làm tử số to hơn so với raw_mse (vì các giá trị đã được scale to lên để tổng=1).
        - Kết quả loss sẽ lớn hơn raw_mse.
        """
        per_row = ((s_dist - t_dist) ** 2 * pair_mask).sum(dim=-1, keepdim=True)
        loss = (per_row * valid_rows).sum() / valid_rows.sum().clamp(min=1.0)
        print(f"Normalized MSE Loss: {loss.item():.8e}")
        return loss

    elif loss_type == "js":
        """
        Jensen-Shannon Divergence:
        - JS divergence rất chuẩn cự ly cho 2 phân phối mảng (tổng=1).
        - JS nằm trong [0, ln(2)] ≈ [0, 0.693]. 
        - JS loss thường có dạng log(phân phối) nên gradient qua JS đều và ổn định hơn MSE cho xác suất.
        """
        mixed = 0.5 * (s_dist + t_dist)
        s_kl = (s_dist * ((s_dist + eps).log() - (mixed + eps).log()) * pair_mask).sum(dim=-1, keepdim=True)
        t_kl = (t_dist * ((t_dist + eps).log() - (mixed + eps).log()) * pair_mask).sum(dim=-1, keepdim=True)
        per_row = 0.5 * (s_kl + t_kl)
        loss = (per_row * valid_rows).sum() / valid_rows.sum().clamp(min=1.0)
        print(f"Jensen-Shannon Loss: {loss.item():.8e}")
        return loss

    else: # KL
        """
        KL Divergence gốc.
        - Khác với JS, KL không có tính đối xứng và có thể vô cùng lớn nếu dự đoán lệch nhiều.
        """
        per_row = (t_dist * ((t_dist + eps).log() - (s_dist + eps).log()) * pair_mask).sum(dim=-1, keepdim=True)
        loss = (per_row * valid_rows).sum() / valid_rows.sum().clamp(min=1.0)
        print(f"KL Divergence Loss: {loss.item():.8e}")
        return loss


if __name__ == "__main__":
    # Test script: tạo 2 tensor ngẫu nhiên mô phỏng attention
    torch.manual_seed(42)
    B = 1 # Batch size
    H = 4 # Heads
    L = 10 # Seq len
    
    # 1. Tạo random logits
    student_logits = torch.randn(B, H, L, L)
    teacher_logits = torch.randn(B, H, L, L)
    
    # Ở teacher, ta làm cho attention có cấu trúc hơn xíu (ví dụ thêm noise lên một ma trận ground truth)
    base_logits = torch.randn(B, H, L, L)
    # Teacher xịn hơn nên sắc nét hơn (nhân temperature)
    teacher_attn = torch.softmax(base_logits * 5.0, dim=-1)
    # Student thì random và kém sắc nét hơn
    student_attn = torch.softmax((base_logits + torch.randn_like(base_logits)*0.5), dim=-1)
    
    # 2. Sinh mask (ví dụ lấy một vùng [L/2:, :L/2])
    pair_mask = torch.zeros(B, 1, L, L, dtype=torch.bool)
    pair_mask[:, :, 5:, :5] = True
    
    print("Thông số tensor:")
    print("Shape:", student_attn.shape)
    print("Student sample probs (row 6):", student_attn[0, 0, 6, :5].tolist())
    print("Teacher sample probs (row 6):", teacher_attn[0, 0, 6, :5].tolist())
    
    # 3. Tính toán từng loại loss
    for lt in ["raw_mse", "mass_mse", "mse", "js", "kl", "cka"]:
        masked_attention_distribution_loss(student_attn, teacher_attn, pair_mask, loss_type=lt)
