# character_recognition

手書き文字認識

## 動作前に

config を設定する。

まずテンプレートからコピー。

```
cp config.template.yml config.yml
```

次に、config.yml を編集。

## MNIST 動作確認

```
python commands/mnist.py
```

## 画像データ生成

```
python commands/generate.py
```

## 認識モデル実験

```
python commands/experiment.py
```

## 認識モデル事前学習

```
python commands/pretrain.py
```

## GUI アプリ

```
python commands/board.py
```
