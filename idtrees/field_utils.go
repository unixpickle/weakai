package idtrees

import (
	"fmt"
	"sort"
)

type BoolFieldGetter func(e Entry) bool
type IntFieldGetter func(e Entry) int
type GeneralFieldGetter func(e Entry) Value

// A ValueAdder adds a Value to an Entry's FieldValues slice.
type ValueAdder func(e Entry, v Value)

// A StringValue is an fmt.Stringer that returns itself.
type StringValue string

// String returns string(s).
func (s StringValue) String() string {
	return string(s)
}

// A BoolValue is an fmt.Stringer that returns "true" or "false"
// depending on its underlying value.
type BoolValue bool

// String returns "true" if b and "false" otherwise.
func (b BoolValue) String() string {
	if b {
		return "true"
	} else {
		return "false"
	}
}

// A BoolField is a field which reports BoolValue(true) and
// BoolValue(false) as its potential values.
type BoolField string

// CreateBoolField generates a field with two possibile answers.
// It adds the field to each Entry in the DataSet using v, specifying
// the boolean value returned by g.
// It returns the generated BoolField.
func CreateBoolField(d *DataSet, g BoolFieldGetter, v ValueAdder, label string) BoolField {
	f := BoolField(label)
	d.Fields = append(d.Fields, f)
	for _, entry := range d.Entries {
		v(entry, BoolValue(g(entry)))
	}
	return f
}

// String returns string(b)
func (b BoolField) String() string {
	return string(b)
}

// Values returns BoolValue(true) and BoolValue(false).
func (b BoolField) Values() []Value {
	return []Value{BoolValue(true), BoolValue(false)}
}

// A ListField is a Field which has a pre-defined
// list of Values.
type ListField struct {
	Label     string
	ValueList []Value
}

// CreateListField generates a field with any number of possible Values.
// It adds the field to each Entry in the DataSet using v, specifying the
// Value returned by g.
// It returns the generated ListField.
func CreateListField(d *DataSet, g GeneralFieldGetter, v ValueAdder, label string) *ListField {
	f := &ListField{Label: label, ValueList: []Value{}}
	d.Fields = append(d.Fields, f)
	seenValues := map[Value]bool{}
	for _, entry := range d.Entries {
		val := g(entry)
		v(entry, val)
		if !seenValues[val] {
			seenValues[val] = true
			f.ValueList = append(f.ValueList, val)
		}
	}
	return f
}

// String returns l.Label.
func (l *ListField) String() string {
	return l.Label
}

// Values returns l.ValueList.
func (l *ListField) Values() []Value {
	return l.ValueList
}

// CreateBisectingIntFields generates zero or more fields which
// divide up integer values associated with each Entry.
// It returns the generated BoolFields, of which there may be none.
//
// The labelFmt argument should be a format string with a "%d" in it.
// This will be used to generate labels for each of the fields.
// It should look something like "X is greater than %d.".
func CreateBisectingIntFields(d *DataSet, g IntFieldGetter, v ValueAdder,
	labelFmt string) []BoolField {
	possibilities := []int{}
	seenPossibilities := map[int]bool{}
	for _, entry := range d.Entries {
		val := g(entry)
		if !seenPossibilities[val] {
			seenPossibilities[val] = true
			possibilities = append(possibilities, val)
		}
	}
	sort.Ints(possibilities)

	if len(possibilities) <= 1 {
		return []BoolField{}
	}

	res := make([]BoolField, 0, len(possibilities)-1)
	for i := 0; i < len(possibilities)-1; i++ {
		middle := (possibilities[i] + possibilities[i+1]) / 2
		field := BoolField(fmt.Sprintf(labelFmt, middle))
		res = append(res, field)
		for _, entry := range d.Entries {
			v(entry, BoolValue(g(entry) > middle))
		}
		d.Fields = append(d.Fields, field)
	}

	return res
}
