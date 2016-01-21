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
	InitialDomain() Domain
}

type Solution map[Variable]Value

type Domain []Value

func DomainIntersection(d1, d2 Domain) Domain {
	res := Domain{}
	for _, x := range d1 {
		if d2.Contains(x) {
			res = append(res, x)
		}
	}
	return res
}

func DomainUnion(d1, d2 Domain) Domain {
	res := make(Domain, len(d1))
	copy(res, d1)
	for _, x := range d2 {
		if !res.Contains(x) {
			res = append(res, x)
		}
	}
	return res
}

func (d Domain) Contains(v Value) bool {
	for _, x := range d {
		if x == v {
			return true
		}
	}
	return false
}
