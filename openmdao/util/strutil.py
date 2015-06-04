
import ast

def get_common_ancestor(name1, name2):
    """
    Returns
    -------
    str
        Absolute name of any common ancestor `System` containing
        both name1 and name2.  If none is found, returns ''.
    """
    common_parts = []
    for part1, part2 in zip(name1.split(':'), name2.split(':')):
        if part1 == part2:
            common_parts.append(part1)
        else:
            break

    if common_parts:
        return ':'.join(common_parts)
    else:
        return ''

class ExprVarScanner(ast.NodeVisitor):
    """
    This node visitor collects all variable names found in the
    AST, and excludes names of functions.  Variables having
    dotted names are not supported.
    """
    def __init__(self):
        self.varnames = set()

    def visit_Name(self, node):
        self.varnames.add(node.id)

    def visit_Call(self, node):
        for arg in node.args:
            self.visit(arg)

    def visit_Attribute(self, node):
        self.visit(node.attr)


def parse_for_vars(expr):
    """
    Parameters
    ----------
    expr : str
        An expression string that we want to parse for variable names.

    Returns
    -------
    list of str
        Names of variables from the given string.
    """
    root = ast.parse(expr, mode='exec')
    scanner = ExprVarScanner()
    scanner.visit(root)
    return scanner.varnames
