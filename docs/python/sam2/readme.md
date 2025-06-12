### SAM2.1

1. [Download ONNX](https://github.com/ryouchinsa/sam-cpp-macos?tab=readme-ov-file)
2. python engine_build.py 
3. python sam_trt.py

``` 运行engine_build.py会生成engine文件 但是目前由于ONNX的属性问题现在只能编译图片大小的engine文件```

```在engine_build.py的line 54中可以修改图片大小的参数```