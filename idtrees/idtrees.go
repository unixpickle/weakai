// Package idtrees facilitates the training of
// identification trees.
package idtrees

// A Sample has a classification and a set of attributes.
type Sample interface {
	// Attr returns the attribute value for the given key.
	//
	// If this returns an int, int64, float32, or float64,
	// then all Samples in the training set must return the
	// same type and the attribute will be used to form
	// splits (e.g. a rule like "x >= 3").
	//
	// If the returned type is not one of the numeric types
	// listed above, then splits are equality based (e.g.
	// a rule like "x == true", or one like "x == Red").
	Attr(name string) interface{}

	// Class returns the class of this Sample.
	// The result may be any type, with the only caveat
	// being that classes must be comparable and thus
	// cannot be map or slices.
	Class() interface{}
}

type Tree struct {
	// Classification is non-nil if this is a leaf,
	// in which case it maps classes to their final
	// probabilities.
	//
	// If the training data was fully separable, this
	// will contain one entry with probability 1.
	Classification map[interface{}]float64

	// If Classification is nil (i.e. this is not a leaf),
	// then this is the attribute used to split the branch.
	// If the attribute refered to by Attr is an int, int64,
	// float32, or float64, then NumSplit is non-nil.
	// If the attribute is not a number type, ValSplit is
	// non-nil.
	Attr string

	NumSplit *NumSplit
	ValSplit ValSplit
}

// NumSplit stores the two branches resulting from
// splitting a tree based on a numerical cutoff.
type NumSplit struct {
	// Threshold is the numerical decision boundary.
	// If a sample's attribute's value is greater
	// than Threshold, then the Greater branch is
	// taken. Otherwise, the LessEqual branch is.
	Threshold interface{}

	LessEqual *Tree
	Greater   *Tree
}

// ValSplit stores the branches resulting from splitting
// a tree by a comparable but non-numeric attribute.
type ValSplit map[interface{}]*Tree
