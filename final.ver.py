import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

# CSV 파일 경로
csv_file = 'C:\\abel\\scope_0.csv'

# CSV 파일 불러오기
df = pd.read_csv(csv_file)

# 첫 번째 행 제외하고 데이터 추출
df = df.iloc[1:, :]

# x 축 데이터는 첫 번째 열로 설정
x = df.iloc[:, 0].astype(float)

# y1 축 데이터는 두 번째 열로 설정
y1 = df.iloc[:, 1].astype(float)

# y2 축 데이터는 세 번째 열로 설정
y2 = df.iloc[:, 2].astype(float)

# STFT 수행
f, t, Zxx_y1 = stft(y1)
f, t, Zxx_y2 = stft(y2)

# STFT 결과 시각화
plt.figure(figsize=(25, 15))

plt.subplot(2, 1, 1)
plt.pcolormesh(t, f, abs(Zxx_y1), shading='nearest')
plt.title('STFT of y1')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 0.2)  # y값 범위 설정
plt.colorbar(label='Magnitude', extend='both')

plt.subplot(2, 1, 2)
plt.pcolormesh(t, f, abs(Zxx_y2), shading='nearest')
plt.title('STFT of y2')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 0.2)  # y값 범위 설정
plt.colorbar(label='Magnitude', extend='both')

plt.tight_layout()
plt.show()
