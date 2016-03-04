<?php
include_once 'Neuron.php';

class Network {

	/**
	 * @var array<Neuron>
	 */
	private $layers = [];

	/**
	 * @var {float}
	 */
	private $learnRate = .4;

	/**
	 * @param $nInputs
	 * @param ...$layerNeurons
	 */
	public function __construct($nInputs, ...$layerNeurons) {
		// Create layers with random neurons
		foreach ($layerNeurons as $index => $nNeurons) {
			$layer = [];
			// Create neurons + 1 bias for layer
			for ($i = 0; $i < $nNeurons; $i++) {
				$layer[$i] = new Neuron($nInputs);
			}
			$this->layers[] = $layer;
			// Update input amount for next layer
			$nInputs = $nNeurons;
		}
	}

	/**
	 * Store the current network weights in a file
	 *
	 * @param $fileName {string}
	 */
	public function save($fileName) {
		file_put_contents($fileName, serialize($this->layers));
	}

	/**
	 * Load network weights from file
	 *
	 * @param $fileName {string}
	 */
	public function load($fileName) {
		if (is_file($fileName)) {
			$layers = unserialize(file_get_contents($fileName));
			if ($layers) {
				$this->layers = $layers;
			}
		}
	}

	/**
	 * @param $inputs
	 * @returns array<float>
	 */
	public function run($inputs) {
		foreach ($this->layers as $layer) {
			$outputs = [];
			// Get output for all neurons on layer
			foreach ($layer as $neuron) {
				$outputs[] = $neuron->output($inputs);
			}
			// Set last outputs as inputs for next layer
			$inputs = $outputs;
		}

		return $outputs;
	}

	/**
	 * Train the network by using backpropagation algorithm
	 *
	 * @param $inputs {Array}
	 * @param $desiredOutput {Array}
	 */
	public function train($inputs, $desiredOutput) {
		$outputs = [$inputs];
		$output = [];
		$currentInput = $inputs;
		// Get outputs of all layers
		foreach ($this->layers as $layer) {
			$output = [];
			// Get output for all neurons on layer
			foreach ($layer as $neuron) {
				$output[] = $neuron->output($currentInput);
			}
			// Set last outputs as inputs for next layer
			$currentInput = $output;
			$outputs[] = $output;
		}
		// Back propagate the errors to weights on all neurons in all layers
		$this->backPropagate($outputs, $desiredOutput);
		// Return the accuracy of the current training
		return $this->getAccuracy($output, $desiredOutput);
	}

	/**
	 * Backpropagation algorithm to calculate and update neuron weights
	 *
	 * @param $outputs {Array}
	 * @param $desired {Array}
	 */
	private function backPropagate($outputs, $desired) {
		$deltas = [];
		// Reverse iterate through layers to calculate neuron deltas
		for ($i = count($this->layers); $i--;) {
			$deltas[$i] = [];
			foreach ($this->layers[$i] as $n => $neuron) {
				$output = $outputs[$i + 1][$n];
				$error = 0;
				if (isset($deltas[$i + 1])) {
					// Sum weight from neuron in current layer to neuron in previous
					// layer multiplied by delta of previous layer
					foreach ($this->layers[$i + 1] as $p => $prevNeuron) {
						$error += $prevNeuron->weights[$n] * $deltas[$i + 1][$p];
					}
				}
				else {
					// Difference from desired and output of current neuron
					// Since $outputs contains also input layer we have 1 more layer
					$error = $desired[$n] - $output;
				}
				// Calculate delta for current neuron
				$deltas[$i][$n] = $output * (1 - $output) * $error;
			}
		}
		// Update the weights of all neurons
		foreach ($this->layers as $l => $layer) {
			foreach ($layer as $n => &$neuron) {
				foreach ($neuron->weights as $w => &$weight) {
					// Adjust weight by delta of this neuron times input for this neuron
					// $outputs contains all initial inputs as additional layer
					// So the layer index is same like input index on $outputs
					// Use original input or bias 1
					$weight += $this->learnRate * $deltas[$l][$n] * (isset($outputs[$l][$w]) ? $outputs[$l][$w] : 0);
				}
				$neuron->bias = $deltas[$l][$n];
			}
		}
	}

	/**
	 * Calculates the accuracy a output has reached
	 *
	 * @param $outputs {array}
	 * @param $desired {array}
	 * @return float
	 */
	private function getAccuracy($outputs, $desired) {
		$accuracy = 0;
		foreach ($outputs as $o => $output) {
			$accuracy += $desired[$o] - $output;
		}

		return 1 - $accuracy / count($outputs);
	}
}