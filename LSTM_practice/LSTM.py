{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "import pickle\n",
    "import time\n",
    "import numpy as np\n",
    "\n",
    "import ptb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sigmoid(x):\n",
    "    return 1 / (1 + np.exp(-x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "###### class LSTM:\n",
    "    def __init__(self, Wx, Wh, b):\n",
    "        '''\n",
    "        Parameters\n",
    "        ----------\n",
    "        Wx: 입력 x에 대한 가중치 매개변수(4개분의 가중치가 담겨 있음)\n",
    "        Wh: 은닉 상태 h에 대한 가중치 매개변수(4개분의 가중치가 담겨 있음)\n",
    "        b: 편향（4개분의 편향이 담겨 있음）\n",
    "        '''\n",
    "        self.params = [Wx, Wh, b]\n",
    "        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]\n",
    "        self.cache = None\n",
    "\n",
    "    def forward(self, x, h_prev, c_prev):\n",
    "        Wx, Wh, b = self.params\n",
    "        N, H = h_prev.shape\n",
    "\n",
    "        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b \n",
    "        # X = N X D, Wx = D X 4H, h_prev = N X H, Wh = H X 4H, A = N X 4H\n",
    "        # N : 미니배치 수\n",
    "        # D : 입력 데이터 차원수\n",
    "        # H : 기억셀과 은닉상태의 차원 수\n",
    "\n",
    "        f = A[:, :H]\n",
    "        g = A[:, H:2*H]\n",
    "        i = A[:, 2*H:3*H]\n",
    "        o = A[:, 3*H:]\n",
    "\n",
    "        f = sigmoid(f)\n",
    "        g = np.tanh(g)\n",
    "        i = sigmoid(i)\n",
    "        o = sigmoid(o)\n",
    "\n",
    "        c_next = f * c_prev + g * i\n",
    "        h_next = o * np.tanh(c_next)\n",
    "\n",
    "        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)\n",
    "        return h_next, c_next\n",
    "\n",
    "    def backward(self, dh_next, dc_next):\n",
    "        Wx, Wh, b = self.params\n",
    "        x, h_prev, c_prev, i, f, g, o, c_next = self.cache\n",
    "\n",
    "        tanh_c_next = np.tanh(c_next)\n",
    "\n",
    "        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)\n",
    "\n",
    "        dc_prev = ds * f\n",
    "\n",
    "        di = ds * g\n",
    "        df = ds * c_prev\n",
    "        do = dh_next * tanh_c_next\n",
    "        dg = ds * i\n",
    "\n",
    "        di *= i * (1 - i)\n",
    "        df *= f * (1 - f)\n",
    "        do *= o * (1 - o)\n",
    "        dg *= (1 - g ** 2)\n",
    "\n",
    "        dA = np.hstack((df, dg, di, do))\n",
    "\n",
    "        dWh = np.dot(h_prev.T, dA)\n",
    "        dWx = np.dot(x.T, dA)\n",
    "        db = dA.sum(axis=0)\n",
    "\n",
    "        self.grads[0][...] = dWx\n",
    "        self.grads[1][...] = dWh\n",
    "        self.grads[2][...] = db\n",
    "\n",
    "        dx = np.dot(dA, Wx.T)\n",
    "        dh_prev = np.dot(dA, Wh.T)\n",
    "\n",
    "        return dx, dh_prev, dc_prev"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "class TimeLSTM:\n",
    "    def __init__(self, Wx, Wh, b, stateful=False):\n",
    "        self.params = [Wx, Wh, b]\n",
    "        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]\n",
    "        self.layers = None\n",
    "\n",
    "        self.h, self.c = None, None\n",
    "        self.dh = None\n",
    "        self.stateful = stateful\n",
    "\n",
    "    def forward(self, xs):\n",
    "        Wx, Wh, b = self.params\n",
    "        N, T, D = xs.shape\n",
    "        H = Wh.shape[0]\n",
    "\n",
    "        self.layers = []\n",
    "        hs = np.empty((N, T, H), dtype='f')\n",
    "\n",
    "        if not self.stateful or self.h is None:\n",
    "            self.h = np.zeros((N, H), dtype='f')\n",
    "        if not self.stateful or self.c is None:\n",
    "            self.c = np.zeros((N, H), dtype='f')\n",
    "\n",
    "        for t in range(T):\n",
    "            layer = LSTM(*self.params)\n",
    "            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)\n",
    "            hs[:, t, :] = self.h\n",
    "\n",
    "            self.layers.append(layer)\n",
    "\n",
    "        return hs\n",
    "\n",
    "    def backward(self, dhs):\n",
    "        Wx, Wh, b = self.params\n",
    "        N, T, H = dhs.shape\n",
    "        D = Wx.shape[0]\n",
    "\n",
    "        dxs = np.empty((N, T, D), dtype='f')\n",
    "        dh, dc = 0, 0\n",
    "\n",
    "        grads = [0, 0, 0]\n",
    "        for t in reversed(range(T)):\n",
    "            layer = self.layers[t]\n",
    "            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)\n",
    "            dxs[:, t, :] = dx\n",
    "            for i, grad in enumerate(layer.grads):\n",
    "                grads[i] += grad\n",
    "\n",
    "        for i, grad in enumerate(grads):\n",
    "            self.grads[i][...] = grad\n",
    "        self.dh = dh\n",
    "        return dxs\n",
    "\n",
    "    def set_state(self, h, c=None):\n",
    "        self.h, self.c = h, c\n",
    "\n",
    "    def reset_state(self):\n",
    "        self.h, self.c = None, None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Embedding:\n",
    "    def __init__(self, W):\n",
    "        self.params = [W]\n",
    "        self.grads = [np.zeros_like(W)]\n",
    "        self.idx = None\n",
    "\n",
    "    def forward(self, idx):\n",
    "        W, = self.params\n",
    "        self.idx = idx\n",
    "        out = W[idx]\n",
    "        return out\n",
    "\n",
    "    def backward(self, dout):\n",
    "        dW, = self.grads\n",
    "        dW[...] = 0\n",
    "        np.add.at(dW, self.idx, dout)\n",
    "        return None\n",
    "\n",
    "class TimeEmbedding:\n",
    "    def __init__(self, W):\n",
    "        self.params = [W]\n",
    "        self.grads = [np.zeros_like(W)]\n",
    "        self.layers = None\n",
    "        self.W = W\n",
    "\n",
    "    def forward(self, xs):\n",
    "        N, T = xs.shape\n",
    "        V, D = self.W.shape\n",
    "\n",
    "        out = np.empty((N, T, D), dtype='f')\n",
    "        self.layers = []\n",
    "\n",
    "        for t in range(T):\n",
    "            layer = Embedding(self.W)\n",
    "            out[:, t, :] = layer.forward(xs[:, t])\n",
    "            self.layers.append(layer)\n",
    "\n",
    "        return out\n",
    "\n",
    "    def backward(self, dout):\n",
    "        N, T, D = dout.shape\n",
    "\n",
    "        grad = 0\n",
    "        for t in range(T):\n",
    "            layer = self.layers[t]\n",
    "            layer.backward(dout[:, t, :])\n",
    "            grad += layer.grads[0]\n",
    "\n",
    "        self.grads[0][...] = grad\n",
    "        return None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "class TimeDropout:\n",
    "    def __init__(self, dropout_ratio=0.5):\n",
    "        self.params, self.grads = [], []\n",
    "        self.dropout_ratio = dropout_ratio\n",
    "        self.mask = None\n",
    "        self.train_flg = True\n",
    "\n",
    "    def forward(self, xs):\n",
    "        if self.train_flg:\n",
    "            flg = np.random.rand(*xs.shape) > self.dropout_ratio\n",
    "            scale = 1 / (1.0 - self.dropout_ratio)\n",
    "            self.mask = flg.astype(np.float32) * scale\n",
    "\n",
    "            return xs * self.mask\n",
    "        else:\n",
    "            return xs\n",
    "\n",
    "    def backward(self, dout):\n",
    "        return dout * self.mask\n",
    "\n",
    "class TimeAffine:\n",
    "    def __init__(self, W, b):\n",
    "        self.params = [W, b]\n",
    "        self.grads = [np.zeros_like(W), np.zeros_like(b)]\n",
    "        self.x = None\n",
    "\n",
    "    def forward(self, x):\n",
    "        N, T, D = x.shape\n",
    "        W, b = self.params\n",
    "\n",
    "        rx = x.reshape(N*T, -1)\n",
    "        out = np.dot(rx, W) + b\n",
    "        self.x = x\n",
    "        return out.reshape(N, T, -1)\n",
    "\n",
    "    def backward(self, dout):\n",
    "        x = self.x\n",
    "        N, T, D = x.shape\n",
    "        W, b = self.params\n",
    "\n",
    "        dout = dout.reshape(N*T, -1)\n",
    "        rx = x.reshape(N*T, -1)\n",
    "\n",
    "        db = np.sum(dout, axis=0)\n",
    "        dW = np.dot(rx.T, dout)\n",
    "        dx = np.dot(dout, W.T)\n",
    "        dx = dx.reshape(*x.shape)\n",
    "\n",
    "        self.grads[0][...] = dW\n",
    "        self.grads[1][...] = db\n",
    "\n",
    "        return dx\n",
    "    \n",
    "def softmax(x):\n",
    "    if x.ndim == 2:\n",
    "        x = x - x.max(axis=1, keepdims=True)\n",
    "        x = np.exp(x)\n",
    "        x /= x.sum(axis=1, keepdims=True)\n",
    "    elif x.ndim == 1:\n",
    "        x = x - np.max(x)\n",
    "        x = np.exp(x) / np.sum(np.exp(x))\n",
    "\n",
    "    return x\n",
    "\n",
    "class TimeSoftmaxWithLoss:\n",
    "    def __init__(self):\n",
    "        self.params, self.grads = [], []\n",
    "        self.cache = None\n",
    "        self.ignore_label = -1\n",
    "\n",
    "    def forward(self, xs, ts):\n",
    "        N, T, V = xs.shape\n",
    "\n",
    "        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우\n",
    "            ts = ts.argmax(axis=2)\n",
    "\n",
    "        mask = (ts != self.ignore_label)\n",
    "\n",
    "        # 배치용과 시계열용을 정리(reshape)\n",
    "        xs = xs.reshape(N * T, V)\n",
    "        ts = ts.reshape(N * T)\n",
    "        mask = mask.reshape(N * T)\n",
    "\n",
    "        ys = softmax(xs)\n",
    "        ls = np.log(ys[np.arange(N * T), ts])\n",
    "        ls *= mask  # ignore_label에 해당하는 데이터는 손실을 0으로 설정\n",
    "        loss = -np.sum(ls)\n",
    "        loss /= mask.sum()\n",
    "\n",
    "        self.cache = (ts, ys, mask, (N, T, V))\n",
    "        return loss\n",
    "\n",
    "    def backward(self, dout=1):\n",
    "        ts, ys, mask, (N, T, V) = self.cache\n",
    "\n",
    "        dx = ys\n",
    "        dx[np.arange(N * T), ts] -= 1\n",
    "        dx *= dout\n",
    "        dx /= mask.sum()\n",
    "        dx *= mask[:, np.newaxis]  # ignore_labelㅇㅔ 해당하는 데이터는 기울기를 0으로 설정\n",
    "\n",
    "        dx = dx.reshape((N, T, V))\n",
    "\n",
    "        return dx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "GPU  = False\n",
    "\n",
    "if GPU:\n",
    "    import cupy as np\n",
    "    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)\n",
    "    np.add.at = np.scatter_add\n",
    "\n",
    "    print('\\033[92m' + '-' * 60 + '\\033[0m')\n",
    "    print(' ' * 23 + '\\033[92mGPU Mode (cupy)\\033[0m')\n",
    "    print('\\033[92m' + '-' * 60 + '\\033[0m\\n')\n",
    "else:\n",
    "    import numpy as np\n",
    "\n",
    "def to_cpu(x):\n",
    "    import numpy\n",
    "    if type(x) == numpy.ndarray:\n",
    "        return x\n",
    "    return np.asnumpy(x)\n",
    "\n",
    "\n",
    "def to_gpu(x):\n",
    "    import cupy\n",
    "    if type(x) == cupy.ndarray:\n",
    "        return x\n",
    "    return cupy.asarray(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "class BaseModel:\n",
    "    def __init__(self):\n",
    "        self.params, self.grads = None, None\n",
    "\n",
    "    def forward(self, *args):\n",
    "        raise NotImplementedError\n",
    "\n",
    "    def backward(self, *args):\n",
    "        raise NotImplementedError\n",
    "\n",
    "    def save_params(self, file_name=None):\n",
    "        if file_name is None:\n",
    "            file_name = self.__class__.__name__ + '.pkl'\n",
    "\n",
    "        params = [p.astype(np.float16) for p in self.params]\n",
    "        if GPU:\n",
    "            params = [to_cpu(p) for p in params]\n",
    "\n",
    "        with open(file_name, 'wb') as f:\n",
    "            pickle.dump(params, f)\n",
    "\n",
    "    def load_params(self, file_name=None):\n",
    "        if file_name is None:\n",
    "            file_name = self.__class__.__name__ + '.pkl'\n",
    "\n",
    "        if '/' in file_name:\n",
    "            file_name = file_name.replace('/', os.sep)\n",
    "\n",
    "        if not os.path.exists(file_name):\n",
    "            raise IOError('No file: ' + file_name)\n",
    "\n",
    "        with open(file_name, 'rb') as f:\n",
    "            params = pickle.load(f)\n",
    "\n",
    "        params = [p.astype('f') for p in params]\n",
    "        if GPU:\n",
    "            params = [to_gpu(p) for p in params]\n",
    "\n",
    "        for i, param in enumerate(self.params):\n",
    "            param[...] = params[i]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "class BetterRnnlm(BaseModel):\n",
    "    '''\n",
    "     LSTM 계층을 2개 사용하고 각 층에 드롭아웃을 적용한 모델이다.\n",
    "     아래 [1]에서 제안한 모델을 기초로 하였고, [2]와 [3]의 가중치 공유(weight tying)를 적용했다.\n",
    "     [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)\n",
    "     [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)\n",
    "     [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)\n",
    "    '''\n",
    "    def __init__(self, vocab_size=10000, wordvec_size=650,\n",
    "                 hidden_size=650, dropout_ratio=0.5):\n",
    "        V, D, H = vocab_size, wordvec_size, hidden_size\n",
    "        rn = np.random.randn\n",
    "\n",
    "        embed_W = (rn(V, D) / 100).astype('f')\n",
    "        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')\n",
    "        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')\n",
    "        lstm_b1 = np.zeros(4 * H).astype('f')\n",
    "        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')\n",
    "        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')\n",
    "        lstm_b2 = np.zeros(4 * H).astype('f')\n",
    "        affine_b = np.zeros(V).astype('f')\n",
    "\n",
    "        self.layers = [\n",
    "            TimeEmbedding(embed_W),\n",
    "            TimeDropout(dropout_ratio),\n",
    "            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),\n",
    "            TimeDropout(dropout_ratio),\n",
    "            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),\n",
    "            TimeDropout(dropout_ratio),\n",
    "            TimeAffine(embed_W.T, affine_b)  # weight tying!!\n",
    "        ]\n",
    "        self.loss_layer = TimeSoftmaxWithLoss()\n",
    "        self.lstm_layers = [self.layers[2], self.layers[4]]\n",
    "        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]\n",
    "\n",
    "        self.params, self.grads = [], []\n",
    "        for layer in self.layers:\n",
    "            self.params += layer.params\n",
    "            self.grads += layer.grads\n",
    "\n",
    "    def predict(self, xs, train_flg=False):\n",
    "        for layer in self.drop_layers:\n",
    "            layer.train_flg = train_flg\n",
    "\n",
    "        for layer in self.layers:\n",
    "            xs = layer.forward(xs)\n",
    "        return xs\n",
    "\n",
    "    def forward(self, xs, ts, train_flg=True):\n",
    "        score = self.predict(xs, train_flg)\n",
    "        loss = self.loss_layer.forward(score, ts)\n",
    "        return loss\n",
    "\n",
    "    def backward(self, dout=1):\n",
    "        dout = self.loss_layer.backward(dout)\n",
    "        for layer in reversed(self.layers):\n",
    "            dout = layer.backward(dout)\n",
    "        return dout\n",
    "\n",
    "    def reset_state(self):\n",
    "        for layer in self.lstm_layers:\n",
    "            layer.reset_state()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "class SGD:\n",
    "    '''\n",
    "    확률적 경사하강법(Stochastic Gradient Descent)\n",
    "    '''\n",
    "    def __init__(self, lr=0.01):\n",
    "        self.lr = lr\n",
    "        \n",
    "    def update(self, params, grads):\n",
    "        for i in range(len(params)):\n",
    "            params[i] -= self.lr * grads[i]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clip_grads(grads, max_norm):\n",
    "    total_norm = 0\n",
    "    for grad in grads:\n",
    "        total_norm += np.sum(grad ** 2)\n",
    "    total_norm = np.sqrt(total_norm)\n",
    "\n",
    "    rate = max_norm / (total_norm + 1e-6)\n",
    "    if rate < 1:\n",
    "        for grad in grads:\n",
    "            grad *= rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_duplicate(params, grads):\n",
    "    '''\n",
    "    매개변수 배열 중 중복되는 가중치를 하나로 모아\n",
    "    그 가중치에 대응하는 기울기를 더한다.\n",
    "    '''\n",
    "    params, grads = params[:], grads[:]  # copy list\n",
    "\n",
    "    while True:\n",
    "        find_flg = False\n",
    "        L = len(params)\n",
    "\n",
    "        for i in range(0, L - 1):\n",
    "            for j in range(i + 1, L):\n",
    "                # 가중치 공유 시\n",
    "                if params[i] is params[j]:\n",
    "                    grads[i] += grads[j]  # 경사를 더함\n",
    "                    find_flg = True\n",
    "                    params.pop(j)\n",
    "                    grads.pop(j)\n",
    "                # 가중치를 전치행렬로 공유하는 경우(weight tying)\n",
    "                elif params[i].ndim == 2 and params[j].ndim == 2 and \\\n",
    "                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):\n",
    "                    grads[i] += grads[j].T\n",
    "                    find_flg = True\n",
    "                    params.pop(j)\n",
    "                    grads.pop(j)\n",
    "\n",
    "                if find_flg: break\n",
    "            if find_flg: break\n",
    "\n",
    "        if not find_flg: break\n",
    "\n",
    "    return params, grads"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "class RnnlmTrainer:\n",
    "    def __init__(self, model, optimizer):\n",
    "        self.model = model\n",
    "        self.optimizer = optimizer\n",
    "        self.time_idx = None\n",
    "        self.ppl_list = None\n",
    "        self.eval_interval = None\n",
    "        self.current_epoch = 0\n",
    "\n",
    "    def get_batch(self, x, t, batch_size, time_size):\n",
    "        batch_x = np.empty((batch_size, time_size), dtype='i')\n",
    "        batch_t = np.empty((batch_size, time_size), dtype='i')\n",
    "\n",
    "        data_size = len(x)\n",
    "        jump = data_size // batch_size\n",
    "        offsets = [i * jump for i in range(batch_size)]  # 배치에서 각 샘플을 읽기 시작하는 위치\n",
    "\n",
    "        for time in range(time_size):\n",
    "            for i, offset in enumerate(offsets):\n",
    "                batch_x[i, time] = x[(offset + self.time_idx) % data_size]\n",
    "                batch_t[i, time] = t[(offset + self.time_idx) % data_size]\n",
    "            self.time_idx += 1\n",
    "        return batch_x, batch_t\n",
    "\n",
    "    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,\n",
    "            max_grad=None, eval_interval=20):\n",
    "        data_size = len(xs)\n",
    "        max_iters = data_size // (batch_size * time_size)\n",
    "        self.time_idx = 0\n",
    "        self.ppl_list = []\n",
    "        self.eval_interval = eval_interval\n",
    "        model, optimizer = self.model, self.optimizer\n",
    "        total_loss = 0\n",
    "        loss_count = 0\n",
    "\n",
    "        start_time = time.time()\n",
    "        for epoch in range(max_epoch):\n",
    "            for iters in range(max_iters):\n",
    "                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)\n",
    "\n",
    "                # 기울기를 구해 매개변수 갱신\n",
    "                loss = model.forward(batch_x, batch_t)\n",
    "                model.backward()\n",
    "                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음\n",
    "                if max_grad is not None:\n",
    "                    clip_grads(grads, max_grad)\n",
    "                optimizer.update(params, grads)\n",
    "                total_loss += loss\n",
    "                loss_count += 1\n",
    "\n",
    "                # 퍼플렉서티 평가\n",
    "                if (eval_interval is not None) and (iters % eval_interval) == 0:\n",
    "                    ppl = np.exp(total_loss / loss_count)\n",
    "                    elapsed_time = time.time() - start_time\n",
    "                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'\n",
    "                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))\n",
    "                    self.ppl_list.append(float(ppl))\n",
    "                    total_loss, loss_count = 0, 0\n",
    "\n",
    "            self.current_epoch += 1\n",
    "\n",
    "    def plot(self, ylim=None):\n",
    "        x = numpy.arange(len(self.ppl_list))\n",
    "        if ylim is not None:\n",
    "            plt.ylim(*ylim)\n",
    "        plt.plot(x, self.ppl_list, label='train')\n",
    "        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')\n",
    "        plt.ylabel('퍼플렉서티')\n",
    "        plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def eval_perplexity(model, corpus, batch_size=10, time_size=35):\n",
    "    print('퍼플렉서티 평가 중 ...')\n",
    "    corpus_size = len(corpus)\n",
    "    total_loss, loss_cnt = 0, 0\n",
    "    max_iters = (corpus_size - 1) // (batch_size * time_size)\n",
    "    jump = (corpus_size - 1) // batch_size\n",
    "\n",
    "    for iters in range(max_iters):\n",
    "        xs = np.zeros((batch_size, time_size), dtype=np.int32)\n",
    "        ts = np.zeros((batch_size, time_size), dtype=np.int32)\n",
    "        time_offset = iters * time_size\n",
    "        offsets = [time_offset + (i * jump) for i in range(batch_size)]\n",
    "        for t in range(time_size):\n",
    "            for i, offset in enumerate(offsets):\n",
    "                xs[i, t] = corpus[(offset + t) % corpus_size]\n",
    "                ts[i, t] = corpus[(offset + t + 1) % corpus_size]\n",
    "\n",
    "        try:\n",
    "            loss = model.forward(xs, ts, train_flg=False)\n",
    "        except TypeError:\n",
    "            loss = model.forward(xs, ts)\n",
    "        total_loss += loss\n",
    "\n",
    "        sys.stdout.write('\\r%d / %d' % (iters, max_iters))\n",
    "        sys.stdout.flush()\n",
    "\n",
    "    print('')\n",
    "    ppl = np.exp(total_loss / max_iters)\n",
    "    return ppl\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "학습 및 평가"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "| 에폭 1 |  반복 1 / 1327 | 시간 1[s] | 퍼플렉서티 10000.16\n",
      "| 에폭 1 |  반복 21 / 1327 | 시간 27[s] | 퍼플렉서티 3316.29\n",
      "| 에폭 1 |  반복 41 / 1327 | 시간 51[s] | 퍼플렉서티 1679.42\n",
      "| 에폭 1 |  반복 61 / 1327 | 시간 75[s] | 퍼플렉서티 1295.72\n",
      "| 에폭 1 |  반복 81 / 1327 | 시간 98[s] | 퍼플렉서티 1079.33\n",
      "| 에폭 1 |  반복 101 / 1327 | 시간 124[s] | 퍼플렉서티 901.92\n",
      "| 에폭 1 |  반복 121 / 1327 | 시간 148[s] | 퍼플렉서티 796.39\n",
      "| 에폭 1 |  반복 141 / 1327 | 시간 172[s] | 퍼플렉서티 711.82\n",
      "| 에폭 1 |  반복 161 / 1327 | 시간 196[s] | 퍼플렉서티 693.78\n",
      "| 에폭 1 |  반복 181 / 1327 | 시간 219[s] | 퍼플렉서티 687.82\n",
      "| 에폭 1 |  반복 201 / 1327 | 시간 243[s] | 퍼플렉서티 605.43\n",
      "| 에폭 1 |  반복 221 / 1327 | 시간 267[s] | 퍼플렉서티 587.32\n",
      "| 에폭 1 |  반복 241 / 1327 | 시간 291[s] | 퍼플렉서티 523.34\n",
      "| 에폭 1 |  반복 261 / 1327 | 시간 315[s] | 퍼플렉서티 545.06\n",
      "| 에폭 1 |  반복 281 / 1327 | 시간 338[s] | 퍼플렉서티 519.47\n",
      "| 에폭 1 |  반복 301 / 1327 | 시간 362[s] | 퍼플렉서티 447.81\n",
      "| 에폭 1 |  반복 321 / 1327 | 시간 385[s] | 퍼플렉서티 395.88\n",
      "| 에폭 1 |  반복 341 / 1327 | 시간 410[s] | 퍼플렉서티 450.28\n",
      "| 에폭 1 |  반복 361 / 1327 | 시간 435[s] | 퍼플렉서티 470.45\n",
      "| 에폭 1 |  반복 381 / 1327 | 시간 460[s] | 퍼플렉서티 388.77\n",
      "| 에폭 1 |  반복 401 / 1327 | 시간 485[s] | 퍼플렉서티 404.20\n",
      "| 에폭 1 |  반복 421 / 1327 | 시간 510[s] | 퍼플렉서티 402.87\n",
      "| 에폭 1 |  반복 441 / 1327 | 시간 534[s] | 퍼플렉서티 380.71\n",
      "| 에폭 1 |  반복 461 / 1327 | 시간 561[s] | 퍼플렉서티 375.63\n",
      "| 에폭 1 |  반복 481 / 1327 | 시간 585[s] | 퍼플렉서티 350.96\n",
      "| 에폭 1 |  반복 501 / 1327 | 시간 609[s] | 퍼플렉서티 350.48\n",
      "| 에폭 1 |  반복 521 / 1327 | 시간 632[s] | 퍼플렉서티 351.11\n",
      "| 에폭 1 |  반복 541 / 1327 | 시간 657[s] | 퍼플렉서티 360.41\n",
      "| 에폭 1 |  반복 561 / 1327 | 시간 680[s] | 퍼플렉서티 321.74\n",
      "| 에폭 1 |  반복 581 / 1327 | 시간 704[s] | 퍼플렉서티 295.91\n",
      "| 에폭 1 |  반복 601 / 1327 | 시간 727[s] | 퍼플렉서티 380.63\n",
      "| 에폭 1 |  반복 621 / 1327 | 시간 753[s] | 퍼플렉서티 349.02\n",
      "| 에폭 1 |  반복 641 / 1327 | 시간 778[s] | 퍼플렉서티 313.86\n",
      "| 에폭 1 |  반복 661 / 1327 | 시간 801[s] | 퍼플렉서티 302.24\n",
      "| 에폭 1 |  반복 681 / 1327 | 시간 824[s] | 퍼플렉서티 257.93\n",
      "| 에폭 1 |  반복 701 / 1327 | 시간 846[s] | 퍼플렉서티 279.94\n",
      "| 에폭 1 |  반복 721 / 1327 | 시간 869[s] | 퍼플렉서티 288.20\n",
      "| 에폭 1 |  반복 741 / 1327 | 시간 891[s] | 퍼플렉서티 246.57\n",
      "| 에폭 1 |  반복 761 / 1327 | 시간 914[s] | 퍼플렉서티 259.85\n",
      "| 에폭 1 |  반복 781 / 1327 | 시간 936[s] | 퍼플렉서티 244.37\n",
      "| 에폭 1 |  반복 801 / 1327 | 시간 959[s] | 퍼플렉서티 269.49\n",
      "| 에폭 1 |  반복 821 / 1327 | 시간 982[s] | 퍼플렉서티 250.95\n",
      "| 에폭 1 |  반복 841 / 1327 | 시간 1005[s] | 퍼플렉서티 256.52\n",
      "| 에폭 1 |  반복 861 / 1327 | 시간 1028[s] | 퍼플렉서티 251.54\n",
      "| 에폭 1 |  반복 881 / 1327 | 시간 1053[s] | 퍼플렉서티 231.78\n",
      "| 에폭 1 |  반복 901 / 1327 | 시간 1077[s] | 퍼플렉서티 283.72\n",
      "| 에폭 1 |  반복 921 / 1327 | 시간 1101[s] | 퍼플렉서티 257.28\n",
      "| 에폭 1 |  반복 941 / 1327 | 시간 1125[s] | 퍼플렉서티 253.75\n",
      "| 에폭 1 |  반복 961 / 1327 | 시간 1149[s] | 퍼플렉서티 273.78\n",
      "| 에폭 1 |  반복 981 / 1327 | 시간 1172[s] | 퍼플렉서티 257.10\n",
      "| 에폭 1 |  반복 1001 / 1327 | 시간 1196[s] | 퍼플렉서티 216.35\n",
      "| 에폭 1 |  반복 1021 / 1327 | 시간 1219[s] | 퍼플렉서티 253.46\n",
      "| 에폭 1 |  반복 1041 / 1327 | 시간 1243[s] | 퍼플렉서티 229.48\n",
      "| 에폭 1 |  반복 1061 / 1327 | 시간 1266[s] | 퍼플렉서티 217.94\n",
      "| 에폭 1 |  반복 1081 / 1327 | 시간 1290[s] | 퍼플렉서티 189.82\n",
      "| 에폭 1 |  반복 1101 / 1327 | 시간 1314[s] | 퍼플렉서티 215.07\n",
      "| 에폭 1 |  반복 1121 / 1327 | 시간 1338[s] | 퍼플렉서티 257.32\n",
      "| 에폭 1 |  반복 1141 / 1327 | 시간 1362[s] | 퍼플렉서티 231.32\n",
      "| 에폭 1 |  반복 1161 / 1327 | 시간 1386[s] | 퍼플렉서티 218.40\n",
      "| 에폭 1 |  반복 1181 / 1327 | 시간 1411[s] | 퍼플렉서티 209.83\n",
      "| 에폭 1 |  반복 1201 / 1327 | 시간 1435[s] | 퍼플렉서티 181.03\n",
      "| 에폭 1 |  반복 1221 / 1327 | 시간 1459[s] | 퍼플렉서티 178.38\n",
      "| 에폭 1 |  반복 1241 / 1327 | 시간 1482[s] | 퍼플렉서티 208.09\n",
      "| 에폭 1 |  반복 1261 / 1327 | 시간 1506[s] | 퍼플렉서티 191.04\n",
      "| 에폭 1 |  반복 1281 / 1327 | 시간 1529[s] | 퍼플렉서티 197.27\n",
      "| 에폭 1 |  반복 1301 / 1327 | 시간 1553[s] | 퍼플렉서티 250.04\n",
      "| 에폭 1 |  반복 1321 / 1327 | 시간 1579[s] | 퍼플렉서티 230.36\n",
      "퍼플렉서티 평가 중 ...\n",
      "209 / 210\n",
      "검증 퍼플렉서티:  197.09913561417102\n",
      "--------------------------------------------------\n",
      "| 에폭 2 |  반복 1 / 1327 | 시간 1[s] | 퍼플렉서티 285.67\n",
      "| 에폭 2 |  반복 21 / 1327 | 시간 26[s] | 퍼플렉서티 226.51\n",
      "| 에폭 2 |  반복 41 / 1327 | 시간 50[s] | 퍼플렉서티 212.46\n",
      "| 에폭 2 |  반복 61 / 1327 | 시간 74[s] | 퍼플렉서티 197.34\n",
      "| 에폭 2 |  반복 81 / 1327 | 시간 97[s] | 퍼플렉서티 179.51\n",
      "| 에폭 2 |  반복 101 / 1327 | 시간 121[s] | 퍼플렉서티 168.49\n",
      "| 에폭 2 |  반복 121 / 1327 | 시간 145[s] | 퍼플렉서티 180.99\n",
      "| 에폭 2 |  반복 141 / 1327 | 시간 169[s] | 퍼플렉서티 199.50\n",
      "| 에폭 2 |  반복 161 / 1327 | 시간 194[s] | 퍼플렉서티 213.96\n",
      "| 에폭 2 |  반복 181 / 1327 | 시간 218[s] | 퍼플렉서티 223.25\n",
      "| 에폭 2 |  반복 201 / 1327 | 시간 242[s] | 퍼플렉서티 206.58\n",
      "| 에폭 2 |  반복 221 / 1327 | 시간 266[s] | 퍼플렉서티 203.46\n",
      "| 에폭 2 |  반복 241 / 1327 | 시간 289[s] | 퍼플렉서티 196.86\n"
     ]
    }
   ],
   "source": [
    "# 하이퍼파라미터 설정\n",
    "batch_size = 20\n",
    "wordvec_size = 650\n",
    "hidden_size = 650\n",
    "time_size = 35\n",
    "lr = 20.0\n",
    "max_epoch = 40\n",
    "max_grad = 0.25\n",
    "dropout = 0.5\n",
    "\n",
    "# 학습 데이터 읽기\n",
    "corpus, word_to_id, id_to_word = ptb.load_data('train')\n",
    "corpus_val, _, _ = ptb.load_data('val')\n",
    "corpus_test, _, _ = ptb.load_data('test')\n",
    "\n",
    "if GPU:\n",
    "    corpus = to_gpu(corpus)\n",
    "    corpus_val = to_gpu(corpus_val)\n",
    "    corpus_test = to_gpu(corpus_test)\n",
    "\n",
    "vocab_size = len(word_to_id)\n",
    "xs = corpus[:-1]\n",
    "ts = corpus[1:]\n",
    "\n",
    "model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)\n",
    "optimizer = SGD(lr)\n",
    "trainer = RnnlmTrainer(model, optimizer)\n",
    "\n",
    "best_ppl = float('inf')\n",
    "for epoch in range(max_epoch):\n",
    "    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,\n",
    "                time_size=time_size, max_grad=max_grad)\n",
    "\n",
    "    model.reset_state()\n",
    "    ppl = eval_perplexity(model, corpus_val)\n",
    "    print('검증 퍼플렉서티: ', ppl)\n",
    "\n",
    "    if best_ppl > ppl:\n",
    "        best_ppl = ppl\n",
    "        model.save_params()\n",
    "    else:\n",
    "        lr /= 4.0\n",
    "        optimizer.lr = lr\n",
    "\n",
    "    model.reset_state()\n",
    "    print('-' * 50)\n",
    "\n",
    "\n",
    "# 테스트 데이터로 평가\n",
    "model.reset_state()\n",
    "ppl_test = eval_perplexity(model, corpus_test)\n",
    "print('테스트 퍼플렉서티: ', ppl_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}