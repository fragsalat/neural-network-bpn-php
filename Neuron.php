<?php

class Neuron {

	/**
	 * @var array<float>
	 */
	public $weights = [];

	public $bias = null;

	/**
	 * initialize a neuron
	 *
	 * @param int $nInputs
	 */
	public function __construct(int $nInputs) {
		for ($i = $nInputs; $i--;) {
			$this->weights[$i] = rand() / getrandmax() * 4 - 2;
		}
		$this->bias = -1;
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

		return $this->activation($sum + $this->bias);
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