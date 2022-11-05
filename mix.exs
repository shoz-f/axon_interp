defmodule AxonInterp.MixProject do
  use Mix.Project

  def project do
    [
      app: :axon_interp,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.4.0"},
      {:exla, "~> 0.4.0"},
      {:axon, "~> 0.3.0"},
      {:axon_onnx, "~> 0.3.0"},
      {:httpoison, "~> 1.8"}
    ]
  end
end
