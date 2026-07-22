import pandas as pd

# 1. 读取原始数据
# 请确保文件名与你本地的文件名一致
# 先在终端运行: pip install openpyxl
import pandas as pd

file_name = 'PA Concrete Database with GWP cement type 5.17.2024.xlsx'
# 读取时指定 Sheet1
df = pd.read_excel(file_name, sheet_name='Sheet1')


# 2. 定义哪些列是“配比信息”
# 我们需要把除了 强度(fc (MPa))、龄期(AGE) 和 强度相关计算(GWP/strength) 之外的所有列作为标识
# 这样具有相同配比但不同 AGE 的行会被合并到同一行
mix_cols = [col for col in df.columns if col not in ['fc (MPa)', 'AGE', 'GWP/strength']]

# 3. 使用 pivot_table 进行转换
# index: 保持不变的配比列
# columns: 要展开成新列的类别（这里是 AGE）
# values: 填入新列数值的来源（这里是强度 fc (MPa)）
df_pivoted = df.pivot_table(index=mix_cols, columns='AGE', values='fc (MPa)').reset_index()

# 4. 重命名新生成的列名
# AGE 列原本的值是 7, 28, 56，转换后会变成列名，我们把它们改得更直观
df_pivoted.rename(columns={7: '7day_strength', 28: '28day_strength', 56: '56day_strength'}, inplace=True)

# 5. 移除列索引层级的名称（可选，为了让表格更干净）
df_pivoted.columns.name = None

# 6. 保存处理后的结果
output_file = 'Concrete_Mix_Pivoted_by_Age.csv'
df_pivoted.to_csv(output_file, index=False)

print(f"处理完成！新表格已保存为: {output_file}")
print("前 5 行预览：")
print(df_pivoted.head())