#!/usr/bin/python3

# Build Markov models of text and generate text from Markov models

# Copyright (c) 2013, William Morrison
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
import random
import itertools
import bisect
import math

class Chain:
    """A tuple-like construct for representing Markov chains of fixed length"""

    def __init__(self, length, seq=None, default=None):
        """Create a new Chain object

        length
            The length of chain. This will be used during creation, where the Chain will be sliced
            or padded appropriately. It is also used in chain.push(), where the chain will not grow
            beyond this length.

        seq
            This is an optional starting sequence for the chain. If it is longer, the last *length*
            elements are used. If it is shorter, the sequence is padded to *length* by prepending
            *default* elements to it.

        default
            This is the default element to use for padding, and defaults to None."""
        
        if not isinstance(length, int) or length < 0:
            raise ValueError("Length must be a non-negative integer.")
        if seq is None:
            self._hist = tuple([default for i in range(length)])
        else:
            if length == 0:
                self._hist = tuple()
            elif len(seq) >= length:
                self._hist = tuple(seq[-length:])
            else:
                self._hist = list([default for i in range(length)])
                self._hist[-len(seq):] = seq
                self._hist = tuple(self._hist)


    def push(self, elem):
        """Return a new Chain with *elem* pushed onto the end. Length is preserved."""
        return Chain(len(self._hist), self._hist+(elem,))

    #################################################
    # Special methods for tuple-like behaviour below.

    def __eq__(self, other):
        return self._hist == other

    def __hash__(self):
        return hash(self._hist)

    def __repr__(self):
        return 'Chain'+repr(self._hist)

    def __iter__(self):
        return iter(self._hist)

    def __getitem__(self, index):
        return self._hist[index]

    def __len__(self):
        return len(self._hist)

def weighted_choice(distribution):
    """Choose an element from a weighted distribution.

    Algorithm is taken from the documentation for the Python random module.

    distribution
        A sequence of (element, weight) tuples."""
    choices, weights = zip(*distribution)
    cdist = list(itertools.accumulate(weights))
    x = random.uniform(0,cdist[-1])
    return choices[bisect.bisect(cdist,x)]
        
def normalize(sequence):
    """Normalize a numeric iterable to a list with the same ratios between numbers, but whose sum
    is 1."""
    
    total=sum(sequence)
    return [elem/total for elem in sequence]

def b_ary_entropy(distribution, base=2):
    """Calculates the entropy of a discrete probability distribution.

    distribution
        A sequence of weights for each element of the distribution.

    base
        The base to calculate the entropy in. E.g., if the base is 2, returns the number of bits of
        entropy. Defaults to base 2."""
    
    return -sum(map(lambda p: p*math.log(p, base), normalize(distribution)))

class Markov:
    """Builds and runs Markov models."""

    def __init__(self, model=None, order=0, end_of_chain=None):
        """Create a new Markov model

        model
            This will create an object that uses an existing Markov model for initial data. If the
            object passed is a Markov object, the order and end_of_chain will be taken from that
            object. If not, the order and end_of_chain values will be taken from the arguments, and
            the model should be a dict from the model property of a Markov object.

        order
            This allows one to change the order of the Markov model that will be created. Markov
            models with higher order keep track of more past events, and are thus larger, but may be
            more accurate if events depends on many past events.

        end_of_chain
            This is the object representing the terminating event. It defaults to None, but if None
            may appear as a non-terminating event, you should set it to something else."""

        if model is not None:
            if isinstance(model, Markov):
                self._order = model.order
                self._table = dict(map(lambda k: (k, model._table[k].copy()), iter(model._table)))
                self._eoc = model.end_of_chain
            else:
                self._order = order
                self._table = dict(map(lambda k: (Chain(len(k), seq=k), model[k].copy()), iter(model)))
                self._eoc = end_of_chain
                
        else:
            if not isinstance(order, int) or order < 0:
                raise ValueError("Markov model order must be a non-negative integer")
            self._order = order
            self._table = {}
            self._eoc = end_of_chain

    def train(self, events):
        """Iterates over events, and updates the internal Markov model with the new data.

        events
            An iterable of events composing one sample Markov chain. Events should be hashable."""

        hist = Chain(self._order)
        
        for event in events:
            self._table[hist][event] = self._table.setdefault(hist, {event:0}).setdefault(event, 0)+1
            hist = hist.push(event)

        self._table[hist][self._eoc] = self._table.setdefault(hist, {self._eoc:0}).setdefault(self._eoc, 0)+1       

    def run(self, chain=None):
        """Run the Markov model, returning a list of events and a count of the entropy inherent in
        the chain.

        chain
            An optional starting point for the model. This must be a Chain object of the same order
            as the model. Events in the chain will not be included in the returned list of
            events."""
        
        if isinstance(chain, Chain) and len(chain) == self._order:
            hist=chain
        else:
            hist = Chain(self._order)
        entropy = 0
        generated = []
        
        while True:
            event = weighted_choice(self._table[hist].items())
            entropy += b_ary_entropy(self._table[hist].values())
            hist = hist.push(event)
            if event == self._eoc:
                break
            generated.append(event)
            
        return generated, entropy

    def __iter__(self):
        """Run the Markov model, yielding a list of events."""

        hist = Chain(self._order)
        while True:
            event = weighted_choice(self._table[hist].items())
            hist = hist.push(event)
            if event == self._eoc:
                break
            yield event

    # expose the order of the model as a property
    @property
    def order(self):
        return self._order

    # expose the termination event as a property
    @property
    def end_of_chain(self):
        return self._eoc

    # allow external objects to get copies of our internal model
    @property
    def model(self):
        return dict(map(lambda k: (tuple(k), self._table[k].copy()), iter(self._table)))
        
# Run this code if this file is invoked as a standalone Python program
if __name__ == "__main__":
    import argparse
    import fileinput
    import json
    import sys

    def dump_model_main(model, fd):
        """Dumps a Markov model's data out to the fd as JSON."""
        table = model.model

        json.dump({
            'distributions':dict(map(lambda k: (hash(k), list(table[k].items())), iter(table))), 
            'chains':dict(map(lambda k: (hash(k), tuple(k)), iter(table))),
            'order':model.order
            }, fd)

    def load_model_main(fd):
        """Loads a JSON file and creates a Markov model from the data."""
        modeldata = json.load(fd)
        table = dict(map(lambda k: (tuple(modeldata['chains'][k]), dict(modeldata['distributions'][k])), 
            iter(modeldata['chains'])))

        return Markov(order=modeldata['order'], model=table)

    def word_iter(lines):
        """Returns chains of word fragments making up words"""
        for line in lines:
            # pull out all "words", here defined as any run of alphabetic characters, or the apostrophe
            samples = list(re.findall("[a-z']+", line.lower()))
            for sample in samples:
                # break the word into groups of letters. These groups are the model's events
                yield re.findall("[aeiouy]+|[bcdfghjklmnpqrstvwxz]+|[']+", sample)

    def sentence_iter(lines):
        """Returns chains of words making up sentences"""
        s = ''.join(lines)       

        samples = list(re.findall("(?:[^\s.]+\s+)*[^\s.]+\.", s, flags=re.MULTILINE))
        for sample in samples:
            # break the word into groups of letters. These groups are the model's events
            yield re.findall("\S+", sample)

    def make_model_main(args):
        m = Markov(order=args.order)
        with fileinput.input(files=args.filenames) as f:
            # break the word into groups of letters. These groups are the model's events
            for chain in args.eventiterator(f):
                m.train(chain)

        dump_model_main(m, sys.stdout)
    
    def run_model_main(args):
        model = load_model_main(args.filename)

        if args.num:
            print(args.chainjoiner.join([args.eventjoiner.join([event for event in model]) for i in range(args.num)]))
            
        elif args.entropy:
            entropy = 0
            words = []
            while entropy < args.entropy:
                events, newentropy = model.run()
                entropy += newentropy
                words.append(args.eventjoiner.join(events))

            print(args.chainjoiner.join(words))
            
    class ChooseChainTypeAction(argparse.Action):
        """This class is used as an action for the argparser, and as a registry
        for the different types of chains supported by the program.
        """
        chainTypes = {
            'words': {
                'chainjoiner':' ',
                'eventjoiner':'',
                'eventiterator':word_iter
                },
            'sentences': {
                'chainjoiner':'\n',
                'eventjoiner':' ',
                'eventiterator':sentence_iter
                },
        }

        def __call__(self, parser, namespace, values, option_string=None):
            for attr in self.chainTypes[values]:
                setattr(namespace, attr, self.chainTypes[values][attr])

    parser = argparse.ArgumentParser(description='Generate and run Markov models of text.')
    subparsers = parser.add_subparsers()

    parser.add_argument('--type', choices=ChooseChainTypeAction.chainTypes.keys(), action=ChooseChainTypeAction, help='Select the type of chain to use')

    # create the parser for the "generate" command
    parser_a = subparsers.add_parser('generate', description='Generate a Markov model of text and output it as JSON')
    parser_a.add_argument('--order', type=int, default=1, help='Order of the model to create. Defaults to %(default)s')
    parser_a.add_argument('filenames', type=str, nargs=argparse.REMAINDER, help='A list of file names containing data that will be used to build the model')
    parser_a.set_defaults(func_main=make_model_main)

    # create the parser for the "run" command
    parser_b = subparsers.add_parser('run', description='Load and run an existing model')
    parser_b.add_argument('filename', type=argparse.FileType('r'), help='The name of a JSON file containing the model to run')
    group_b = parser_b.add_mutually_exclusive_group(required=True)
    group_b.add_argument('--num', type=int, help='Generate and output NUM chains')
    group_b.add_argument('--entropy', type=int, help='Generate chains until ENTROPY bits of entropy have been used. This is intended for creating passwords.')
    parser_b.set_defaults(func_main=run_model_main)
    
    parser.set_defaults(chainjoiner='\n', eventjoiner=' ', eventiterator=sentence_iter)

    args = parser.parse_args()

    args.func_main(args)
