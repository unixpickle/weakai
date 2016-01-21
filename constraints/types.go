package main

type Value interface{}

type Constraint interface {
	// Vairables returns an ordered pair of Variables.
	Variables() []Variable

	// RestrictedDomain computes the domain for one of the
	// two variables (specified by idx) given the domain of
	// the other variable.
	RestrictedDomain(idx int, d []Value) []Value
}

type Variable interface {
	Constraints() []Constraint
	InitialDomain() []Value
}
