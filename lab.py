"""
LISP Interpreter
"""

#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)



#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    # intialize an empty string
    final_str = ""
    for char in source:
        if char in ("(", ")"):
            if char == "(":
                final_str += " ( "
            else:
                final_str += " ) "
        # account for new line
        elif char == "\n":
            final_str += " \n "

        else:
            final_str += char

    # before removing comments
    with_comments = final_str.split(" ")

    without_comments = []
    comment = False
    # go through and remove info that is contained within comments
    for item in with_comments:
        if item == ";":
            comment = True

        if comment:
            if item == "\n":
                comment = False
            else:
                continue
        else:
            if item not in (" ", "", "\n"):
                without_comments.append(item)

    return without_comments


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    # check if invalid token
    if len(tokens) == 0:
        raise SchemeSyntaxError

    # recursive helper
    def parse_helper(index):
        current_token = tokens[index]
        # if not wrapped in parentheses
        if current_token not in ("(", ")"):
            return number_or_symbol(current_token), index + 1
        # check if we have a (
        if current_token == "(":
            recursive_result = []
            next_index = index + 1
            if next_index >= len(tokens):
                raise SchemeSyntaxError
            current_token = tokens[next_index]
            # keep searching the expression until it closes
            while current_token != ")":
                parsed_expression, next_index = parse_helper(next_index)
                recursive_result.append(parsed_expression)
                # if too far
                if next_index >= len(tokens):
                    raise SchemeSyntaxError
                current_token = tokens[next_index]
            return recursive_result, next_index + 1

        elif current_token == ")":
            raise SchemeSyntaxError

    # start with index as 1
    parse_expression, needed_index = parse_helper(0)
    if needed_index != len(tokens):
        raise SchemeSyntaxError
    return parse_expression


######################
# Built-in Functions #
######################
def multiply(args):
    """
    multiplies all the elements of a list
    returns product
    """
    product = 1
    for val in args:
        product *= val
    return product


def divide(args):
    """
    divides all of the elements of a list
    returns quotient
    """
    return args[0] / multiply(args[1:])


def equal(args):
    """
    returns true if all of the elemnts are equal
    and false if they are not
    """

    # checks if all are equal
    ans = args[0]
    for arg in args:
        if arg != ans:
            return False
    return True


def greater_than(args):
    """
    returns true if the arguments are in decreasing order
    false if else
    """
    # check for repeats
    if len(set(args)) != len(args):
        return False
    args_to_sort = args[:]
    args_to_sort.sort(reverse=True)
    return args_to_sort == args


def less_than(args):
    """
    returns true if the args are in increasing order
    false if else
    """
    # make sure they are in correct order
    if len(set(args)) != len(args):
        return False
    args_to_sort = args[:]
    args_to_sort.sort(reverse=False)
    return args_to_sort == args


def greater_than_or_eqal(args):
    """
    returns True if less than or equal to and false otherwise
    """
    for i in range(len(args) - 1):
        if args[i] < args[i + 1]:
            return False
    return True


def not_func(args):
    """
    Handles the not case
    """
    # if statement is false return true
    if len(args) != 1:
        raise SchemeEvaluationError
    if args[0]:
        return False
    return True


def less_than_or_equal(args):
    """
    returns True if less than or equal to and false otherwise
    """
    for i in range(len(args) - 1):
        if args[i] > args[i + 1]:
            return False
    return True


def cons(args):
    """
    creates an instance of pair
    """
    if len(args) != 2:
        raise SchemeEvaluationError
    pair_obj = Pair(args[0], args[1])
    return pair_obj


def car_finder(args):
    """
    gets the first value
    """
    if len(args) == 1 and isinstance(args[0], Pair):
        return args[0].car
    raise SchemeEvaluationError


def cdr_finder(args):
    """
    gets the second value
    """
    if len(args) == 1 and isinstance(args[0], Pair):
        return args[0].cdr
    raise SchemeEvaluationError


def list_func(args):
    """
    creates a list object
    """
    if len(args) == 0:
        return None
    if len(args) == 1:
        return Pair(args[0], None)
    else:
        return Pair(args[0], list_func(args[1:]))


def is_list(args):
    """
    returns True if a list false if not
    """
    current_list = args[0]
    if current_list is None:
        return True
    if not isinstance(current_list, Pair):
        return False
    return is_list([current_list.cdr])


def len_list(args):
    """
    returns the length of the list
    """
    if is_list(args):
        count = 0
        current_arg = args[0]
        # keep track of how many cars
        while current_arg is not None:
            count += 1
            current_arg = current_arg.cdr
        return count
    raise SchemeEvaluationError


def list_ref(args, current_index=0):
    """
    returns item in list at a given index
    """
    wanted_list = args[0]
    index = args[1]
    if not is_list([wanted_list]) and isinstance(wanted_list, Pair) and index == 0:
        return wanted_list.car
    elif not is_list([wanted_list]):
        raise SchemeEvaluationError
    if wanted_list is None:
        raise SchemeEvaluationError
    if current_index == index and isinstance(wanted_list, Pair):
        return wanted_list.car
    if is_list([wanted_list]):
        current_index += 1
        wanted_list = wanted_list.cdr
        return list_ref([wanted_list, index], current_index)


def append_list(args):
    """
    appends list items into one list that
    is a shallow copy
    """
    for arg in args:
        if is_list([arg]):
            continue
        raise SchemeEvaluationError
    new_args_list = [arg for arg in args if arg is not None]
    if len(new_args_list) == 0:
        return None

    # helper function to create shallow copy
    def make_copy(ll):
        if ll is None:
            return None
        else:
            copy_list = Pair(ll.car, make_copy(ll.cdr))
            return copy_list

    # check if recursive and continue to add to list
    if len(new_args_list) == 1:
        return make_copy(new_args_list[0])
    else:
        first_list = make_copy(new_args_list[0])
        end = first_list
        for i in range(1, len(new_args_list)):
            while end.cdr is not None:
                end = end.cdr
            end.cdr = make_copy(new_args_list[i])
        return first_list


def map(args):
    """
    returns a new list where elements are from
    original list but with a function applied
    """

    func = args[0]
    input_list = args[1]
    if callable(func) and is_list([input_list]):
        if input_list is None:
            return None
        # recursively build list
        final_list = Pair(func([input_list.car]), map([func, input_list.cdr]))
        return final_list

    raise SchemeEvaluationError


def filter(args):
    """
    returns new list with just list objects
    that are true with func
    """

    func = args[0]
    input_list = args[1]
    if callable(func) and is_list([input_list]):
        if input_list is None:
            return None
        val = func([input_list.car])
        if val:
            # recursively build list
            final_list = Pair(input_list.car, filter([func, input_list.cdr]))
            return final_list
        else:
            return filter([func, input_list.cdr])
    raise SchemeEvaluationError


def reduce(args):
    """
    takes list func and inital value
    and generates output by applying func with
    intial value to successive elements
    """
    func = args[0]
    input_list = args[1]
    initial_val = args[2]
    if callable(func) and is_list([input_list]):
        if input_list is None:
            # if gone through list return val
            return initial_val
        # keep track of what the val is
        val = func([initial_val, input_list.car])
        return reduce([func, input_list.cdr, val])
    raise SchemeEvaluationError


def begin(args):
    """
    returns the last argument
    """
    return args[-1]


scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": multiply,
    "/": divide,
    "equal?": equal,
    "<": less_than,
    ">": greater_than,
    ">=": greater_than_or_eqal,
    "<=": less_than_or_equal,
    "#t": True,
    "#f": False,
    "not": not_func,
    "cons": cons,
    "car": car_finder,
    "cdr": cdr_finder,
    "nil": None,
    "list": list_func,
    "list?": is_list,
    "length": len_list,
    "list-ref": list_ref,
    "append": append_list,
    "map": map,
    "reduce": reduce,
    "filter": filter,
    "begin": begin,
}


##############
# Evaluation #
##############


class Frames:
    """
    Creates frame object to track what frame we
    are in and what is contained in that frame
    and if there is a parent frame
    """

    def __init__(self, variables, parent):
        self.variables = variables  # this would expect a dict input
        self.parent = parent  # expects frame object

    def assign_variable(self, variable_name, value):
        self.variables[variable_name] = value
        # updates dict

    def find_variable_value(self, variable_name):
        current_frame = self.variables
        # if in the first frame return the value assigned to it at that spot
        if variable_name in current_frame:
            return current_frame[variable_name]
        # if not in first frame should check parent frames
        else:
            if self.parent is not None:
                return self.parent.find_variable_value(variable_name)
            raise SchemeNameError

    def reassign_variable(self, variable_name, value):
        # finds frame and reassigns variable in the frame
        current_frame = self.variables
        if variable_name in current_frame:
            current_frame[variable_name] = value
        else:
            if self.parent is not None:
                return self.parent.reassign_variable(variable_name, value)
            raise SchemeNameError


class LambdaFunc:
    """
    handles and creates function objects
    keeps track of how to call func and
    all necessary info in the func
    """

    def __init__(self, parameters, code_body, enclosing_frame):
        self.parameters = parameters  # this should be a list
        self.code_body = code_body  # this should be a list
        self.enclosing_frame = enclosing_frame  # frame in which defined

    def __call__(self, parameter_vals):
        # check for error
        if len(parameter_vals) != len(self.parameters):
            raise SchemeEvaluationError
        count = 0
        # create variable map
        param_matching = {}
        for param in self.parameters:
            param_matching[param] = parameter_vals[count]
            count += 1
        # evaluate func
        current_frame = Frames(param_matching, self.enclosing_frame)
        return evaluate(self.code_body, current_frame)


class Pair:
    def __init__(self, first_element, second_element):
        self.car = first_element
        self.cdr = second_element

    def __str__(self):
        ans_1 = self.car
        ans_2 = self.cdr
        return "list is " + str(ans_1) + ", " + str(ans_2)


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # intialize a frame
    if frame is None:
        frame = Frames({}, None)
        frame.parent = Frames(scheme_builtins, None)

    # handle different types of expressions
    # if var or num
    if isinstance(tree, str):
        return frame.find_variable_value(tree)
    if isinstance(tree, (int, float)):
        return tree

    else:
        if len(tree) == 0:
            raise SchemeEvaluationError
        # if a func or variable
        if tree[0] == "define":
            # func
            if isinstance(tree[1], list):
                func_name = tree[1][0]
                func_parameters = tree[1][1:]
                func_code = tree[2]
                func = LambdaFunc(func_parameters, func_code, frame)
                frame.assign_variable(func_name, func)
                return func
            # var
            else:
                recursive_result = evaluate(tree[2], frame)
                frame.assign_variable(tree[1], recursive_result)
                return recursive_result

        # here check if the first word is lambda
        if tree[0] == "lambda":
            new_func = LambdaFunc(tree[1], tree[2], frame)
            return new_func

        # here check if first word is an if statement and handle it
        if tree[0] == "if":
            predicate, true_express, false_express = tree[1], tree[2], tree[3]
            pred_ans = evaluate(predicate, frame)
            if pred_ans:
                return evaluate(true_express, frame)
            return evaluate(false_express, frame)
        # handles the and case
        if tree[0] == "and":
            expressions = tree[1:]
            for expression in expressions:
                if evaluate(expression, frame):
                    continue
                return False
            return True
        # handles the or case
        if tree[0] == "or":
            expressions = tree[1:]
            for expression in expressions:
                if evaluate(expression, frame):
                    return True
            return False
        # handles the del case
        if tree[0] == "del":
            variable = tree[1]
            variable_dict = frame.variables
            if variable in variable_dict:
                value = variable_dict.pop(variable)
                return value
            raise SchemeNameError
        # handles the let case
        if tree[0] == "let":
            var_bindings = tree[1]
            body = tree[2]
            bindings_dict = {}
            # create variable bindings for new frame
            for var_bind in var_bindings:
                value = evaluate(var_bind[1], frame)
                bindings_dict[var_bind[0]] = value
            new_frame = Frames(bindings_dict, frame)
            return evaluate(body, new_frame)

        # handles the set! case
        if tree[0] == "set!":
            # frame class should be able to search and reassgin variable
            variable_name = tree[1]
            expression = evaluate(tree[2], frame)
            # new frame object
            frame.reassign_variable(variable_name, expression)
            return expression

        current_operation = evaluate(tree[0], frame)
        if isinstance(current_operation, int):
            raise SchemeEvaluationError
        to_apply = []

        for item in tree[1:]:
            to_apply.append(evaluate(item, frame))
        return current_operation(to_apply)


def result_and_frame(tree, frame=None):
    """
    returns the result of evaluate and the frame
    """
    if frame is None:
        frame = Frames({}, None)
        frame.parent = Frames(scheme_builtins, None)

    evaluate_ans = evaluate(tree, frame)

    return (evaluate_ans, frame)


def evaluate_file(file_name, frame=None):
    """
    returns the result of evaluating a file
    """
    # is this what we should use ?
    with open(file_name) as file_evaluate:
        code = file_evaluate.read()
    return evaluate(parse(tokenize(code)), frame)


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    import os

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import schemerepl

    schemerepl.SchemeREPL(use_frames=True, verbose=False).cmdloop()
