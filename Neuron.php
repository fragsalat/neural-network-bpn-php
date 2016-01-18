<?php

class Neuron {

	/**
	 * @var array<float>
	 */
	public $weights = [];

	/**
	 * initialize a neuron
	 *
	 * @param int $nInputs
	 */
	public function __construct(int $nInputs) {
		for ($i = $nInputs + 1; $i--;) {
			$this->weights[$i] = rand() / getrandmax() * 4 - 2;
		}
	}

	/**
	 * Get the output of a neuron
	 *
	 * @param array $inputs
	 * @return float
	 */
	public function output(array $inputs) {
		$sum = 0;
		foreach ($inputs as $index => $input) {
			$sum += $input * $this->weights[$index];
		}
		//$sum += end($this->weights) * 1;
		return $this->activation($sum);
	}

	/**
	 * Activation function of this neuron: sigmoid
	 *
	 * @param $value float
	 * @return float
	 */
	public function activation(float $value) {
		return  1 / (1 + exp(-$value));
	}
}