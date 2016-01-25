package idtrees

import "fmt"

type Value fmt.Stringer

// A Field is analogous to a question in an identification tree.
// A field might be something like "What color is the monkey's fur?".
// The corresponding values indicate the possible answers to the question.
type Field interface {
	fmt.Stringer

	Values() []Value
}

// An Entry represents an entity that has exactly one Value for each Field.
// Each Entry is a member of exactly one class, which is represented as a Value.
type Entry interface {
	FieldValues() []Value
	Class() Value
}
