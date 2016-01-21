package main

const updateNeighbors = true
const updateReducedNeighborsNeighbors = true
const rotateValues = true

func SolveConstraints(vars []Variable, cancelChan <-chan struct{}) <-chan Solution {
	solutionChan := make(chan Solution, 1)
	go solveConstraints(solverParams{vars, cancelChan, solutionChan})
	return solutionChan
}

type solverParams struct {
	variables  []Variable
	cancelChan <-chan struct{}
	solutions  chan<- Solution
}

func solveConstraints(p solverParams) {
	defer close(p.solutions)

	start := &partialSolution{
		domains:         map[Variable]Domain{},
		currentVariable: 0,
		allValues:       Domain{},
		rotationIndex:   0,
	}

	for _, variable := range p.variables {
		start.domains[variable] = variable.InitialDomain()
		start.allValues = DomainUnion(start.allValues, variable.InitialDomain())
	}

	solveConstraintsStep(start, p)
}

func solveConstraintsStep(sol *partialSolution, p solverParams) bool {
	if sol.currentVariable == len(p.variables) {
		solution := Solution{}
		for variable, domain := range sol.domains {
			solution[variable] = domain[0]
		}
		select {
		case p.solutions <- solution:
			return true
		case <-p.cancelChan:
			return false
		}
	}

	select {
	case <-p.cancelChan:
		return false
	default:
	}

	startValueIndex := sol.rotationIndex
	currentValueIndex := startValueIndex
	for {
		value := sol.allValues[currentValueIndex]
		variable := p.variables[sol.currentVariable]

		newSol := sol.Assign(variable, value)
		if newSol != nil {
			newSol.currentVariable++
			if rotateValues {
				newSol.rotationIndex = (currentValueIndex + 1) % len(sol.allValues)
			}
			if !solveConstraintsStep(newSol, p) {
				return false
			}
		}

		currentValueIndex = (currentValueIndex + 1) % len(sol.allValues)
		if currentValueIndex == startValueIndex {
			break
		}
	}

	return true
}

type partialSolution struct {
	domains map[Variable]Domain

	currentVariable int

	allValues     Domain
	rotationIndex int
}

func (p *partialSolution) Assign(variable Variable, value Value) *partialSolution {
	if !p.domains[variable].Contains(value) {
		return nil
	}
	res := *p
	res.domains = map[Variable]Domain{}
	for key, domain := range p.domains {
		if key == variable {
			res.domains[key] = Domain{value}
		} else {
			res.domains[key] = domain
		}
	}
	if !res.constrainVariable(variable, updateNeighbors) {
		return nil
	}
	return &res
}

func (p *partialSolution) constrainVariable(v Variable, forceExpand bool) bool {
	domain := p.domains[v]
	oldCount := len(domain)
	for _, constraint := range v.Constraints() {
		vars := constraint.Variables()
		var subDomain Domain
		if vars[0] == v {
			otherDomain := p.domains[vars[1]]
			subDomain = constraint.RestrictedDomain(0, otherDomain)
		} else {
			otherDomain := p.domains[vars[0]]
			subDomain = constraint.RestrictedDomain(1, otherDomain)
		}
		domain = DomainIntersection(domain, subDomain)
	}
	p.domains[v] = domain

	if len(domain) == 0 {
		return false
	}

	reducedToOne := len(domain) == 1 && oldCount > 1
	if forceExpand || (updateReducedNeighborsNeighbors && reducedToOne) {
		for _, constraint := range v.Constraints() {
			vars := constraint.Variables()
			var otherVariable Variable
			if vars[0] == v {
				otherVariable = vars[1]
			} else {
				otherVariable = vars[0]
			}
			if !p.constrainVariable(otherVariable, false) {
				return false
			}
		}
	}

	return true
}
