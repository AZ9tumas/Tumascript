
#######################################
# IMPORTS
#######################################

import string

#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# TOKENS
#######################################

TT_INT		= 'INT'
TT_FLOAT    = 'FLOAT'
TT_PLUS     = 'PLUS'
TT_MINUS    = 'MINUS'
TT_MUL      = 'MUL'
TT_DIV      = 'DIV'
TT_MOD      = 'MOD'
TT_LPAREN   = 'LPAREN'
TT_RPAREN   = 'RPAREN'
TT_STRING   = 'STRING'
TT_POW      = 'POW'
TT_EQUALS   = 'EQUAL'
TT_EOF      = 'EOF'
TT_KEYWORD  = 'KEYWORD'
TT_IDNTIFR  = 'IDENTIFIER'

KEYWORDS = (
    "var"
)

class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Error():

    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{colors.FAIL}{self.error_name}: {self.details}{colors.ENDC}\n'
        result += f'{colors.WARNING}File {self.pos_start.fn}; Line: {self.pos_start.ln + 1}{colors.ENDC}'
        return result

class Token:
    def __init__(self, TT, TV=None, pos_start = None, pos_end = None):
        self.tokentype = TT
        self.tokenValue = TV

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end
        
    def __repr__(self):
        return f'{self.tokentype}:{self.tokenValue}' if self.tokenValue != None else str(self.tokentype)

class Position:
    def __init__(self,idx,ln,col,fn,ftext):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftext = ftext

    def advance(self, current_char=None):
        self.idx += 1
        self.col +=1 

        if current_char == '/n':
            self.ln += 1
            self.col = 0
        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftext)

#######################################
# LEXER
#######################################

class Lexer:
    def __init__(self, fn, code):
        self.fn = fn
        self.code = code
        self.pos = Position(-1,0,-1,self.fn, self.code)
        self.current_char = None
        self.advance()
        
    def advance(self, times = 1):
        
        for i in range(0,times):
            self.pos.advance(self.current_char)
            self.current_char = self.code[self.pos.idx] if self.pos.idx < len(self.code) else None
        
    def Make_Tokens(self):
        
        tokens = []
        
        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == '=':
                tokens.append(Token(TT_EQUALS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                if self.match(1, '//'):
                    self.advance()
                    self.make_comment()
                    continue
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '%':
                tokens.append(Token(TT_MOD, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], Error(pos_start, self.pos, 'IllegalCharacterError', f"'{char}'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def match(self, end, expected):
        original = self.code[self.pos.idx:end+1]
        return expected == original

    def make_comment(self):
        self.advance()
        comment = ''
        while self.current_char != '/' and self.match(1, '//') != True:
            if self.current_char == None: return
            comment += self.current_char
            self.advance()
        self.advance(2)
        return

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()
        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    def make_identifier(self):
        identifier = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
            identifier += self.current_char
            self.advance()
        
        tok_type = TT_KEYWORD if identifier in KEYWORDS else TT_IDNTIFR
        return Token(tok_type, identifier, pos_start, self.pos)


#######################################
# NODES
#######################################

class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'

class BinOpNode:
    def __init__(self, left_node, op, right_node):
        self.left_node = left_node
        self.op = op
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op}, {self.right_node})'

class UnaryOpNode:
    def __init__(self, op, node):
        self.op = op
        self.node = node
        self.pos_start = op.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op}, {self.node})'

#######################################
# PARSE RESULT
#######################################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node

        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


#######################################
# PARSER
#######################################

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]

        return self.current_tok

    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.tokentype != TT_EOF:
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', "Expected '+', '-', '/', '*'"))
        return res

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.tokentype in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        elif tok.tokentype in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.tokentype == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.tokentype == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', "Expected ')' to close '('"))

        return res.failure(Error(tok.pos_start, tok.pos_end, 'InvalidSyntaxError', 'Expected INT or FLOAT'))

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV, TT_MOD, TT_POW))

    def expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def bin_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error: return res

        while self.current_tok.tokentype in ops:
            op = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error: return res
            left = BinOpNode(left, op, right)
        return res.success(left)

#######################################
# VALUES
#######################################

class Number:
    def __init__(self, value):
        self.value = value
        self.error = None

        self.set_pos()

    def set_pos(self, pos_start = None, pos_end = None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value)

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value)

    def multiplied_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value)
    
    def divided_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                result = Number(0)
                result.error = Error(self.pos_start, self.pos_end, 'Division by 0', f'Attempt to divide {self.value} by 0')
                return result
            return Number(self.value / other.value)

    def modded_by(self, other):
        if isinstance(other, Number):
            return Number(self.value % other.value)

    def power_of(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value)

    def __repr__(self):
        return str(self.value)

#######################################
# INTERPRETER
#######################################

class Interpreter:
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node)

    def no_visit_method(self, node):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    #######################################

    def visit_NumberNode(self, node):
        return Number(node.tok.tokenValue).set_pos(node.pos_start, node.pos_end)
    
    def visit_BinOpNode(self, node):
        
        left = self.visit(node.left_node)
        right = self.visit(node.right_node)

        if left.error: return left
        if right.error: return right

        if node.op.tokentype == TT_PLUS:
            result = left.added_to(right)
        elif node.op.tokentype == TT_MINUS:
            result = left.subbed_by(right)
        elif node.op.tokentype == TT_DIV:
            result = left.divided_by(right)
        elif node.op.tokentype == TT_MUL:
            result = left.multiplied_by(right)
        elif node.op.tokentype == TT_MOD:
            result = left.modded_by(right)
        elif node.op.tokentype == TT_POW:
            result = left.power_of(right)

        if result.error: return result
        return result.set_pos(node.pos_start, node.pos_end)

    def visit_UnaryOpNode(self, node):
        number = self.visit(node.node)
        
        if node.op.tokentype == TT_MINUS:
            number = number.multiplied_by(Number(-1))

            return number.set_pos(node.pos_start, node.pos_end)


#######################################
# RUN
#######################################
def run(fn, code):
    lexer = Lexer(fn, code)
    tokens, error = lexer.Make_Tokens()
    if error: return None, error

    print(tokens)
    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # Run program
    interpreter = Interpreter()
    result = interpreter.visit(ast.node)
    if result.error: return None, result.error

    return result, None
