# Oscilloscope Viewer

オシロスコープの保存波形を読み込み、表示・画像保存・FWHM計算を行うPyQtアプリです。

## 対応機種

- LeCroy WaveJet 354A
- Rohde & Schwarz RTE1204

画面上部の `Oscilloscope` から機種を手動選択してからファイルを開きます。

### LeCroy WaveJet 354A

従来形式のCSVを1ファイル選択します。`Ch1 V` から始まる列ヘッダーより前をメタデータ、後ろを波形として読み込みます。

### Rohde & Schwarz RTE1204

同じ名前で出力される次の2ファイルを同じフォルダに置きます。

```text
measurement.csv
measurement.Wfm.csv
```

どちらを選択しても、もう一方を自動的に探して読み込みます。

## 開発環境

```powershell
uv sync
uv run oscilloscope.py
uv run pytest
```

## Windows exeの作成

```powershell
.\build_exe.cmd
```

完成したファイルは `dist\OscilloscopeViewer.exe` に作成されます。`--onefile` 形式なので配布は簡単ですが、起動はPython環境から実行する場合より少し遅くなります。
