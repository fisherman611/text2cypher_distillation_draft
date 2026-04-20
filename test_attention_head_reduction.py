import torch

# ---------------------------------------------------------
# BƯỚC 1: Thu thập Raw Attention (Tạo số liệu giả định)
# ---------------------------------------------------------
Batch = 1
Seq_len = 3
H_student = 2  # Student có 2 Heads
H_teacher = 4  # Teacher có 4 Heads

# Giả sử Attention Weights lấy từ Hook (đã qua Softmax tổng = 1)
# Shape: [Batch, Heads, Length, Length]
student_attn = torch.rand(Batch, 2, 3, 3) 
teacher_attn = torch.rand(Batch, 4, 3, 3) 

print(f"1. Ban đầu:")
print(f" - Shape Student Attention: {student_attn.shape}")
print(f" - Shape Teacher Attention: {teacher_attn.shape}")

# ---------------------------------------------------------
# BƯỚC 2: Gọi hàm quy chuẩn align_attention_heads
# ---------------------------------------------------------
# Do 2 heads != 4 heads, nếu args=auto hoặc mean, hàm sẽ kích hoạt tính Toán Trung Bình (Mean Reduction)
def align_attention_heads_mock(s_attn, t_attn):
    # Lấy trung bình ở chiều Head (dim=1)
    s_mean = s_attn.mean(dim=1, keepdim=True)
    t_mean = t_attn.mean(dim=1, keepdim=True)
    return s_mean, t_mean

student_attn_aligned, teacher_attn_aligned = align_attention_heads_mock(student_attn, teacher_attn)

print(f"\n2. Sau khi chạy align_attention_heads (Reduction = mean):")
print(f" - Shape Student mới: {student_attn_aligned.shape}  (Đã nén 2 heads thành 1)")
print(f" - Shape Teacher mới: {teacher_attn_aligned.shape}  (Đã nén 4 heads thành 1)")

"""
Giải thích nội bộ hàm Mean:
Giả sử Token 3 nhìn vào Token 1:
- Đầu (Head) số 1 của Teacher cho xác suất: 0.1
- Đầu (Head) số 2 của Teacher cho xác suất: 0.8  <-- Head này chuyên chú ý điểm này
- Đầu (Head) số 3 của Teacher cho xác suất: 0.05
- Đầu (Head) số 4 của Teacher cho xác suất: 0.05
=> Sau khi Reduction = Mean, giá trị lưu vào ma trận đại diện chung của Teacher sẽ là:
(0.1 + 0.8 + 0.05 + 0.05) / 4 = 0.25
"""

# ---------------------------------------------------------
# BƯỚC 3: Tạo Pair Mask (Chỉ tập trung vào vùng cần tính loss)
# ---------------------------------------------------------
# Giả sử ta chỉ muốn ép học sinh bắt chước cách Token 3 nhìn vào Token 1 (như Schema tới Query)
pair_mask = torch.zeros(Batch, 1, Seq_len, Seq_len, dtype=torch.bool)
pair_mask[0, 0, 2, 0] = True  # Row index 2 (Token 3), Col index 0 (Token 1)

print(f"\n3. Tạo Mask lọc vùng cần quan tâm:")
print(f" - Mask Shape: {pair_mask.shape}")
print(f" - Tổng số pixel được học (Sum mask): {pair_mask.sum().item()}")

# ---------------------------------------------------------
# BƯỚC 4: Tính Loss (Ví dụ: tính Raw MSE)
# ---------------------------------------------------------
# Chỉ so sánh sự khác biệt tại đúng ô mà mask = True
diff_square = ((student_attn_aligned.float() - teacher_attn_aligned.float()) ** 2)
diff_masked = diff_square * pair_mask 

# Tử số: Tổng các chênh lệch
numerator = diff_masked.sum()
# Mẫu số: Tổng số ô học sinh phải học
denominator = pair_mask.sum().clamp(min=1.0) 

loss_mse = numerator / denominator

print(f"\n4. Kết quả Loss Final:")
print(f" - Raw MSE Loss: {loss_mse.item():.6f}")
