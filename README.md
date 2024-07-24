# Lyra
可计算分子谐振频率与电声耦合

## 安装
给`lyra.py`加上可执行权限：
```bash
chmod +x lyra.py
```
把`lyra.py`所在路径添加进`$PATH`变量。

## 使用方法
### 查看帮助
```bash
lyra.py --help
lyra.py freq --help
lyra.py evc --help
```

### 频率计算
需提供Gaussian/Q-Chem频率计算得到的`.fchk`文件，执行
```bash
lyra.py freq xxx.fchk
```

### 电声耦合（以及Huang-Rhys因子、重组能）
如果选择位移谐振子近似/垂直梯度近似，需提供基态频率计算任务和激发态梯度计算任务的`.fchk`文件，执行
```bash
lyra.py evc ground_state.fchk --VG excited_state.fchk
```

如果选择绝热Hessian方法（就是没有近似的），需提供初态和末态计算任务的`.fchk`文件，执行
```bash
lyra.py evc initial_state.fchk --AH final_state.fchk
```

