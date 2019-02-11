import {
	random,
	multiply,
	dotMultiply,
	mean,
	abs,
	subtract,
	transpose,
	add
} from 'mathjs';
import * as activation from './activations';

export class NeuralNetwork {
	constructor(...args) {
		this.input_nodes = args[0]; //number of input neurons
		this.hidden_nodes = args[1]; //number of hidden neurons
		this.output_nodes = args[2]; //number of output neurons

		this.epochs = 50000;
		this.activation = activation.sigmoid;
		this.lr = 0.5; //learning rate
		this.output = 0;

		this.synapse0 = random([this.input_nodes, this.hidden_nodes], -1.0, 1.0); //connections from input layer to hiden
		this.synapse1 = random([this.hidden_nodes, this.output_nodes], -1.0, 1.0); //connections from hidden layer to output
	}

	train(X, y) {
		for (let i = 0; i < this.epochs; i++) {
			// forward prop
			let a1 = X;
			let a2 = multiply(a1, this.synapse0).map(x => this.activation(x, false));
			let a3 = multiply(a2, this.synapse1).map(x => this.activation(x, false));

			// back prop
			let del3 = subtract(y, a3);
			// delta3 = del3*f'(a3)
			let delta3 = dotMultiply(del3, a3.map(x => this.activation(x, true)));
			let del2 = multiply(delta3, transpose(this.synapse1));
			// delta2 = del2*f'(a2)
			let delta2 = dotMultiply(del2, a2.map(x => this.activation(x, true)));

			// gradient descent
			this.synapse1 = add(
				this.synapse1,
				multiply(transpose(a2), multiply(delta3, this.lr))
			);
			this.synapse0 = add(
				this.synapse0,
				multiply(transpose(a1), multiply(delta2, this.lr))
			);
			if (i % 10000 == 0) console.log(`Error: ${mean(abs(del3))}`);
		}
	}

	predict(input) {
		let a1 = input;
		let a2 = multiply(a1, this.synapse0).map(v => this.activation(v, false));
		let a3 = multiply(a2, this.synapse1).map(v => this.activation(v, false));
		return a3;
	}
}
