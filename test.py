import torch
from models.DDNet_Original_ODE import DDNet_Original_NODEFull  # sửa lại nếu tên file khác

# Các thông số giả định (giống config huấn luyện)
frame_l = 64
joint_n = 25
joint_d = 3
feat_d = 150
filters = 64
class_num = 1

# Khởi tạo mô hình
model = DDNet_Original_NODEFull(
    frame_l=frame_l,
    joint_n=joint_n,
    joint_d=joint_d,
    feat_d=feat_d,
    filters=filters,
    class_num=class_num
)

# Chuyển sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Tạo đầu vào giả
M = torch.randn(2, frame_l, feat_d).to(device)
P = torch.randn(2, frame_l, joint_n, joint_d).to(device)

# Chạy forward
with torch.no_grad():
    out = model(M, P)

print("✅ Forward thành công!")
print("Output shape:", out.shape)  # dự kiến: [2, class_num]
