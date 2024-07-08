# 假设已经读取了数据并计算了序列长度，如果没有，可以重新读取数据并计算长度
# data <- read.csv("/Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/分好的数据集csv/Zm/sorted_data.csv", header = TRUE, stringsAsFactors = FALSE)
# data$sequence_length <- nchar(data$sequence)

# 过滤长度小于10000的序列
short_sequences <- data[data$sequence_length < 10000, ]

# 统计标签为1和0的比例
label_counts <- table(short_sequences$label)
percent_label1 <- label_counts[1] / sum(label_counts) * 100
percent_label0 <- label_counts[2] / sum(label_counts) * 100
# 计算长度小于10000的序列数
short_sequences_count <- sum(data$sequence_length < 10000)
# 计算小于10000的序列占总数的百分比
percent_short_sequences <- short_sequences_count / nrow(data) * 100
# 输出结果
cat("长度小于10000的序列占总序列数的百分比:", percent_short_sequences, "%\n")

# 输出长度小于10000的序列中标签为1和0的比例
cat("长度小于10000的序列中:\n")
cat("小于10000的序列个数:", nrow(short_sequences), "\n")
cat("标签为1的个数:", label_counts[1], "\n")
cat("标签为0的个数:", label_counts[2], "\n")
cat("标签为1的比例:", percent_label1, "%\n")
cat("标签为0的比例:", percent_label0, "%\n")

# 计算整体数据集中标签为1和0的比例
total_label_counts <- table(data$label)
total_percent_label1 <- label_counts[1] / sum(total_label_counts[1]) * 100
total_percent_label0 <- label_counts[2] / sum(total_label_counts[2]) * 100

# 输出整体数据集中标签为1和0的比例
cat("\n整体数据集中:\n")
cat("整体序列个数:", nrow(data), "\n")
cat("标签为1的个数:", total_label_counts[1], "\n")
cat("标签为0的个数:", total_label_counts[2], "\n")
cat("标签为1的比例:", total_percent_label1, "%\n")
cat("标签为0的比例:", total_percent_label0, "%\n")




# 计算 Pearson 相关系数
correlation <- cor(data$sequence_length, data$label)

# 输出 Pearson 相关系数及其显著性
cat("Pearson相关系数:", correlation, "\n")

# 计算相关系数的显著性
cor_test <- cor.test(data$sequence_length, data$label)
cat("p 值:", cor_test$p.value, "\n")


# library(ggplot2)
# # 拟合 logistic 回归模型
# model <- glm(label ~ sequence_length, data = data, family = binomial)

# # 创建用于绘制预测曲线的新数据
# new_data <- data.frame(sequence_length = seq(min(data$sequence_length), max(data$sequence_length), length.out = 100))

# # 预测值和置信区间
# new_data$predicted <- predict(model, new_data, type = "response")

# # 绘制图形
# p <- ggplot(data, aes(x = sequence_length, y = label)) +
#   geom_point(alpha = 0.6) +  # 绘制实际数据点
#   geom_line(data = new_data, aes(y = predicted), color = "blue", size = 1) +  # 绘制回归曲线
#   labs(title = "Logistic 回归曲线", x = "序列长度", y = "标签（概率）") +
#   theme_minimal()

# print(p)