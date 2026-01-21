import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体支持（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取训练日志
log_file = "training_log.txt"

print("正在读取训练日志...")

try:
    # 使用pandas读取CSV
    df = pd.read_csv(log_file)
    print("✅ 成功读取日志文件")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())
    
    epochs = df['Epoch'].values
    train_loss = df['Train_Loss'].values
    val_score = df['Val_Score'].values
    
except Exception as e:
    print(f"❌ pandas读取失败: {e}")
    print("尝试手动解析...")
    
    # 手动解析
    epochs = []
    train_loss = []
    val_score = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"总共读取到 {len(lines)} 行")
        
        # 跳过表头（第一行）
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            if line:
                try:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        epochs.append(int(parts[0]))
                        train_loss.append(float(parts[1]))
                        val_score.append(float(parts[2]))
                    else:
                        print(f"⚠️ 第{i+1}行列数不足: {line}")
                except Exception as parse_error:
                    print(f"⚠️ 解析第{i+1}行失败: {parse_error}, 内容: {line}")
    
    epochs = np.array(epochs)
    train_loss = np.array(train_loss)
    val_score = np.array(val_score)

print(f"\n📊 解析完成: {len(epochs)} 个epoch的数据")

# 绘制图表
plt.figure(figsize=(14, 6))

# 子图1: Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'r-', linewidth=2.5, label='Train Loss')  # 去掉 'o'，只用实线
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
# 设置x轴刻度为每20个单位一个刻度
max_epoch = max(epochs)
epoch_ticks = list(range(0, max_epoch + 1, 100))  # 从0开始，每20一个刻度
plt.xticks(epoch_ticks)

# 子图2: Dice Score
plt.subplot(1, 2, 2)
plt.plot(epochs, val_score, 'b-', linewidth=2.5, label='Validation Dice Score')  # 去掉 'o'，只用实线
plt.title('Validation Dice Score Over Epochs', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Dice Score', fontsize=12, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')
# 设置x轴刻度为每20个单位一个刻度
plt.xticks(epoch_ticks)

# 找出最佳性能点（保留但不显示标注文本，只保留红点标记）
best_val_idx = np.argmax(val_score)
best_epoch = epochs[best_val_idx]
best_score = val_score[best_val_idx]

# 在图上标注最佳点（只显示红点，不显示箭头和文本）
plt.subplot(1, 2, 2)
plt.scatter([best_epoch], [best_score], color='red', s=100, zorder=5)
# 注释掉了带箭头的文本标注
# plt.annotate(f'Best: {best_score:.3f}', 
#              xy=(best_epoch, best_score), 
#              xytext=(best_epoch+0.5, best_score+0.02),
#              arrowprops=dict(arrowstyle='->', color='red'),
#              fontsize=10, color='red', fontweight='bold')

plt.tight_layout()

# 保存高质量图片
save_path = 'training_curves_freezing.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n💾 图表已保存为: {save_path}")

# 显示图表
plt.show()

# 输出统计信息
print(f"\n📈 训练统计:")
print(f"总训练轮数: {len(epochs)}")
print(f"初始训练损失: {train_loss[0]:.4f}")
print(f"最终训练损失: {train_loss[-1]:.4f}")
print(f"损失变化: {train_loss[-1] - train_loss[0]:+.4f}")
print(f"最终验证分数: {val_score[-1]:.4f}")
print(f"验证分数提升: {val_score[-1] - val_score[0]:+.4f}")

# 检查是否过拟合
if train_loss[-1] < train_loss[0] and val_score[-1] < val_score[0]:
    print("⚠️  可能出现过拟合：训练损失下降但验证分数也下降")
elif val_score[-1] > val_score[0]:
    print("✅ 模型训练正常：验证分数整体呈上升趋势")