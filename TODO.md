# CT Program 開発

## 5/30

### memo

- resizeプログラムを複数枚の同倍率に対応
- 三好さんからワイヤー入りのファントムをもらう

### TODO

- 複数枚のサイノグラムからスライス画像セットを作るプログラムに着手
  - 中央ぞろえが課題
  - 改良案
    - 試しに一枚作成し，centering_sino.py内のcenter変数を記憶？(プリント?？)
    - centerの値を他のサイノグラムでも使いまわす?

## 6/4

### memo2

- Astra toolboxをできれば使いたい
- 適用方法が問題
  - 現在の流れ
    - Projection --> Sinogram --> Fixed_Sino --> Reconstruction
    - Fixed_SinoをAstra toolboxに適用できればいい
    - とりあえずリファレンスを見ると，Sinogramを作る関数の返り値はint or (int, numpy.ndarray)
      - int は Projection の ID らしい
      - つまり Sinogram データは，np.ndarray で提供すればよさそう