# StencilAcc

## Run

```
mkdir build & cd build
cmake ..
make
./bin/app
```

## src里文件名的说明

第一个数字代表优化的步骤，假如开头是a代表abandon的失败尝试

第二个数字代表大步骤里的小步骤

第三个字母 X 或者 Y代表 nvvp 是否有kernel结果 (NotPass 还是 Pass)