import { exp } from 'mathjs';

export const sigmoid = (x, derivative = false) => {
	let fx = 1 / (1 + exp(-x));
	if (derivative) {
		return fx * (1 - fx);
	}
	return fx;
};
