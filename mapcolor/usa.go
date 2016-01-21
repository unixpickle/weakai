package main

var AdjacentUSAStates = map[string][]string{
	"AK": {},
	"AL": {"TN", "GA", "FL", "MS"},
	"AR": {"MO", "TN", "MS", "LA", "TX", "OK"},
	"AZ": {"UT", "CO", "NM", "CA", "NV"},
	"CA": {"OR", "NV", "AZ", "HI"},
	"CO": {"WY", "NE", "KS", "OK", "NM", "AZ", "UT"},
	"CT": {"MA", "RI", "NY"},
	"DC": {"MD", "VA"},
	"DE": {"PA", "NJ", "MD"},
	"FL": {"GA", "AL"},
	"GA": {"NC", "SC", "FL", "AL", "TN"},
	"HI": {},
	"IA": {"MN", "WI", "IL", "MO", "NE", "SD"},
	"ID": {"MT", "WY", "UT", "NV", "OR", "WA"},
	"IL": {"WI", "IN", "KY", "MO", "IA"},
	"IN": {"MI", "OH", "KY", "IL"},
	"KS": {"NE", "MO", "OK", "CO"},
	"KY": {"OH", "WV", "VA", "TN", "MO", "IL", "IN"},
	"LA": {"AR", "MS", "TX"},
	"MA": {"NH", "RI", "CT", "NY", "VT"},
	"MD": {"PA", "DE", "DC", "VA", "WV"},
	"ME": {"NH"},
	"MI": {"OH", "IN", "WI"},
	"MN": {"WI", "IA", "SD", "ND"},
	"MO": {"IA", "IL", "KY", "TN", "AR", "OK", "KS", "NE"},
	"MS": {"TN", "AL", "LA", "AR"},
	"MT": {"ND", "SD", "WY", "ID"},
	"NC": {"VA", "SC", "GA", "TN"},
	"ND": {"MN", "SD", "MT"},
	"NE": {"SD", "IA", "MO", "KS", "CO", "WY"},
	"NH": {"ME", "MA", "VT"},
	"NJ": {"NY", "DE", "PA"},
	"NM": {"CO", "OK", "TX", "AZ", "UT"},
	"NV": {"ID", "UT", "AZ", "CA", "OR"},
	"NY": {"VT", "MA", "CT", "NJ", "PA"},
	"OH": {"PA", "WV", "KY", "IN", "MI"},
	"OK": {"KS", "MO", "AR", "TX", "NM", "CO"},
	"OR": {"WA", "ID", "NV", "CA"},
	"PA": {"NY", "NJ", "DE", "MD", "WV", "OH"},
	"RI": {"MA", "CT"},
	"SC": {"NC", "GA"},
	"SD": {"ND", "MN", "IA", "NE", "WY", "MT"},
	"TN": {"KY", "VA", "NC", "GA", "AL", "MS", "AR", "MO"},
	"TX": {"OK", "AR", "LA", "NM"},
	"UT": {"ID", "WY", "CO", "NM", "AZ", "NV"},
	"VA": {"MD", "DC", "NC", "TN", "KY", "WV"},
	"VT": {"NH", "MA", "NY"},
	"WA": {"AK", "ID", "OR"},
	"WI": {"MI", "IL", "IA", "MN"},
	"WV": {"PA", "MD", "VA", "KY", "OH"},
	"WY": {"MT", "SD", "NE", "CO", "UT", "ID"},
}

type StateVariable struct {
	Name        string
	constraints []Constraint
}

func (s *StateVariable) Constraints() []Constraint {
	return s.constraints
}

func (s *StateVariable) InitialDomain() Domain {
	return Domain{"#65bcd4", "#9b59b6", "#e36a8f", "#33be6e"}
}

type StateConstraint []Variable

func (c StateConstraint) Variables() []Variable {
	return c
}

func (c StateConstraint) RestrictedDomain(idx int, d Domain) Domain {
	if len(d) != 1 {
		return c[idx].InitialDomain()
	}
	res := Domain{}
	for _, x := range c[idx].InitialDomain() {
		if !d.Contains(x) {
			res = append(res, x)
		}
	}
	return res
}

func VariablesForStates() []Variable {
	vars := map[string]Variable{}
	result := make([]Variable, 0, len(AdjacentUSAStates))
	for stateName := range AdjacentUSAStates {
		v := &StateVariable{Name: stateName, constraints: []Constraint{}}
		vars[stateName] = v
		result = append(result, v)
	}
	for stateName, connections := range AdjacentUSAStates {
		state := vars[stateName]
		for _, connection := range connections {
			c := StateConstraint{state, vars[connection]}
			state.(*StateVariable).constraints = append(state.(*StateVariable).constraints, c)
		}
	}
	return result
}
