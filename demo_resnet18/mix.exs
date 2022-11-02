defmodule DemoR18.MixProject do
  use Mix.Project

  def project do
    [
      app: :demo_r18,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {DemoR18.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    #System.put_env("NNINTERP", "Axon")
    [
      {:nx, "~> 0.4.0"},
      {:exla, "~> 0.4.0"},
      {:cimg, "~> 0.1.13"},
      {:axon_interp, path: ".."}
    ]
  end
end
