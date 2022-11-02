defmodule AxonInterpTest do
  use ExUnit.Case
  #doctest AxonInterp

  test "greets the world" do
    assert AxonInterp.hello() == :world
  end
end
