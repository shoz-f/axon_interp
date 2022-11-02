Mix.install([
  {:axon_onnx, "~> 0.3.0"}
])

AxonOnnx.import("data/resnet18.onnx")
|> then(fn {model, params} -> Axon.serialize(model, params) end)
|> (&File.write("data/resnet18.axon", &1)).()
