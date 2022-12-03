defmodule AxonInterp.MixProject do
  use Mix.Project

  def project do
    [
      app: :axon_interp,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps(),

      # Docs
      # name: "axon_interp",
      source_url: "https://github.com/shoz-f/axon_interp.git",

      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :ssl, :inets]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.4.0"},
      {:exla, "~> 0.4.0"},
      {:axon, "~> 0.3.0"},
      {:axon_onnx, "~> 0.3.0"},
      {:castore, "~> 0.1.19"},

      {:ex_doc, "~> 0.29.1", only: :dev, runtime: false}
    ]
  end

  defp description() do
    "Axon thin wrapper for inference."
  end

  defp package() do
    [
       name: "axon_interp",
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/shoz-f/axon_interp.git"},
      files: ~w(lib mix.exs README* CHANGELOG* LICENSE*)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
#        "LICENSE",
        "CHANGELOG.md",

        #Examples
        "demo_resnet18/Resnet18.livemd",
        "demo_yolov4/YOLOv4.livemd"
      ],
      groups_for_extras: [
        "Examples": Path.wildcard("demo_*/*.livemd")
      ],
#      source_ref: "v#{@version}",
#      source_url: @source_url,
#      skip_undefined_reference_warnings_on: ["CHANGELOG.md"]
    ]
  end
end
