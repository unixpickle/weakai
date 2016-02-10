package svm

// A SubgradientSolver solves Problems using sub-gradient descent.
type SubgradientSolver struct {
	// Tradeoff specifies how important it is to minimize the magnitude of the normal vector versus
	// finding a good separation of samples.
	// In other words, it determines how important a wide margin is.
	// For linearly separable data, you should use a small (but non-zero) Tradeoff value.
	Tradeoff float64

	// Steps indicates how many descents the solver should make before returning its solution.
	// Increasing the number of steps will increase the accuracy, but by decreasing amounts.
	Steps int

	// StepSize is a number between 0 and 1 which determines how much of the gradient should be
	// added to the current solution at each step.
	// Values closer to 0 will result in better accuracy, while values closer to 1 will cause the
	// solver to approach the solution in fewer steps.
	StepSize float64
}

func (s *SubgradientSolver) Solve(p *Problem) *Classifier {
	args := softMarginArgs{
		normal: make(Sample, len(p.Positives[0])),
	}

	for i := 0; i < s.Steps; i++ {
		args = s.descend(p, args)
	}

	return &Classifier{
		HyperplaneNormal: args.normal,
		Threshold:        args.threshold,
		Kernel:           p.Kernel,
	}
}

func (s *SubgradientSolver) descend(p *Problem, args softMarginArgs) softMarginArgs {
	res := args
	res.normal = make(Sample, len(args.normal))
	copy(res.normal, args.normal)

	res.threshold += s.thresholdPartial(p, args) * s.StepSize
	for i := range res.normal {
		res.normal[i] += s.normalPartial(p, args, i) * s.StepSize
	}

	return res
}

// thresholdPartial approximates the partial differential of the soft-margin function with respect
// to the threshold argument.
func (s *SubgradientSolver) thresholdPartial(p *Problem, args softMarginArgs) float64 {
	// TODO: figure out a good "differential" value.
	differential := 1.0 / 10000.0

	tempArgs := args
	tempArgs.threshold += differential
	return (s.softMarginFunction(p, tempArgs) - s.softMarginFunction(p, args)) / differential
}

// normalPartial approximates the partial differential of the soft-margin function with respect to
// a component of the normal vector.
func (s *SubgradientSolver) normalPartial(p *Problem, args softMarginArgs, comp int) float64 {
	// TODO: figure out a good "differential" value.
	differential := 1.0 / 10000.0

	tempArgs := args
	tempArgs.normal = make(Sample, len(args.normal))
	copy(tempArgs.normal, args.normal)
	tempArgs.normal[comp] += differential
	return (s.softMarginFunction(p, tempArgs) - s.softMarginFunction(p, args)) / differential
}

func (s *SubgradientSolver) softMarginFunction(p *Problem, args softMarginArgs) float64 {
	var matchSum float64
	for _, positive := range p.Positives {
		matchSum += 1 - (p.Kernel(args.normal, positive) + args.threshold)
	}
	return matchSum + s.Tradeoff*p.Kernel(args.normal, args.normal)
}

type softMarginArgs struct {
	normal    Sample
	threshold float64
}
