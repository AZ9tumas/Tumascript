
#######################################
# IMPORTS
#######################################
from os import sendfile
import string
from string_with_arrows import *
import math

#######################################
# CONSTANTS
#######################################

LOCKED_VARS = ['null', 'true', 'false', 'print', 'type']

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# TOKENS
#######################################

#INT or FLOAT
TT_INT		= 'INT'
TT_FLOAT    = 'FLOAT'
#Operators
TT_PLUS     = 'PLUS'
TT_MINUS    = 'MINUS'
TT_MUL      = 'MUL'
TT_DIV      = 'DIV'
TT_MOD      = 'MOD'
TT_POW      = 'POW'
TT_EQUALS   = 'EQUAL'
#Round Brackets
TT_LPAREN   = 'LPAREN'
TT_RPAREN   = 'RPAREN'
#String
TT_STRING   = 'STRING'
#Keywords, identifiers
TT_KEYWORD  = 'KEYWORD'
TT_IDNTIFR  = 'IDENTIFIER'

#Boolean operators
TT_GREATER_THAN = 'GREATER_THAN'
TT_SMALLER_THAN = 'SMALLER_THAN'
TT_GREATER_THAN_EQUAL_TO = 'GREATER_THAN_EQUAL_TO'
TT_SMALLER_THAN_EQUAL_TO = 'SMALLER_THAN_EQUAL_TO'
TT_ISEQUAL = 'ISEQUAL'
TT_NOT = 'NOT'

#Other
TT_DOT = 'DOT'
TT_COMMA = 'COMMA'
TT_ARROW = 'ARROW'

#EOF
TT_EOF = 'EOF'

#KEYWORDS
KEYWORDS = [
    "var",
    "not",
    "and",
    "or",
    "if",
    "then",
    "elseif",
    "else",
    "for",
    "while",
    "do",
    "end",
    "func"
]

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

class Error:

    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{colors.FAIL}{self.error_name}: {self.details}{colors.ENDC}\n'
        result += f'{colors.WARNING}File {self.pos_start.fn}; Line: {self.pos_start.ln + 1}{colors.ENDC}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftext, self.pos_start, self.pos_end)
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

    def matches(self, type_, value_):
        return self.tokentype == type_ and self.tokenValue == value_
        
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
                pos_start = self.pos.copy()
                self.advance()
                if self.current_char == '=':
                    tokens.append(Token(TT_ISEQUAL, pos_start=pos_start, pos_end=self.pos))
                    self.advance()
                else:
                    tokens.append(Token(TT_EQUALS, pos_start=self.pos))

            elif self.current_char == '.':
                tokens.append(Token(TT_DOT, pos_start=self.pos))
                self.advance()
            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                pos_start = self.pos.copy()
                not_equals_token = self.make_not_equals()
                if not_equals_token == None:
                    return [], Error(pos_start,self.pos, 'InvalidSyntaxError', "'=' after '!'")
                else:
                    tokens.append(not_equals_token)

            elif self.current_char == '>':
                pos_start = self.pos.copy()
                self.advance()
                if self.current_char == '=':
                    tokens.append(Token(TT_GREATER_THAN_EQUAL_TO, pos_start=pos_start, pos_end=self.pos))
                    self.advance()
                else:
                    tokens.append(Token(TT_GREATER_THAN, pos_start=self.pos))

            elif self.current_char == '<':
                pos_start = self.pos.copy()
                self.advance()
                if self.current_char == '=':
                    tokens.append(Token(TT_SMALLER_THAN_EQUAL_TO, pos_start=pos_start, pos_end=self.pos))
                    self.advance()
                else:
                    tokens.append(Token(TT_SMALLER_THAN, pos_start=self.pos))
                
            
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                self.advance()
                if self.current_char == '>':
                    tokens.append(Token(TT_ARROW, pos_start=self.pos))
                    self.advance()
                else:
                    tokens.append(Token(TT_MINUS, pos_start=self.pos))

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

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(TT_NOT, pos_start=pos_start, pos_end=self.pos)
        self.advance()
        return None

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

class VarAccessNode:
    def __init__(self, var_name_token):
        self.var_name_token = var_name_token

        self.pos_start = var_name_token.pos_start if isinstance(var_name_token, Token) else None
        self.pos_end = var_name_token.pos_end if isinstance(var_name_token, Token) else None

class VarAssignNode:
    def __init__(self, var_name_token, valueNode=None):
        self.var_name_token = var_name_token
        self.valueNode = valueNode

        self.pos_start = var_name_token.pos_start
        self.pos_end = valueNode.pos_end if valueNode != None else self.pos_start

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case
        
        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases) - 1][0]).pos_end

class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, increment, body_node):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.increment = increment
        self.body_node = body_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end

class WhileNode:
    def __init__(self, condition_node, body_node):
        self.condition_node = condition_node
        self.body_node = body_node

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end

class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_tokens, body_node):
        self.var_name_tok = var_name_tok
        self.arg_name_tokens = arg_name_tokens
        self.body_node = body_node

        if self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_tokens) > 0:
            self.pos_start = self.arg_name_tokens[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start
        
        self.pos_end = self.body_node.pos_end

class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start

        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes)-1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end




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

    def __repr__(self):
        return f'{self.node}'

#######################################
# PARSER
#######################################

class Parser:
    def __init__(self, tokens, context):
        self.tokens = tokens
        self.tok_idx = -1
        self.context = context
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]

        return self.current_tok

    def parse(self):
        res = self.expr()
        #print(self.current_tok)
        #print('RESULT: ',res)
        if not res.error and self.current_tok.tokentype != TT_EOF:
            #print(res.node)
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', "Expected '+', '-', '/', '*' lol nub"))
        return res

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(TT_KEYWORD, 'if'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'if'"))
        res.register(self.advance())

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'then'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'then'"))

        res.register(self.advance())
        expr = res.register(self.expr())
        if res.error: return res

        cases.append((condition, expr))

        while self.current_tok.matches(TT_KEYWORD, 'elseif'):
            res.register(self.advance())
            condition = res.register(self.expr())
            if res.error: return res

            if not self.current_tok.matches(TT_KEYWORD, 'then'):
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'then'"))
            res.register(self.advance())

            expr = res.register(self.expr())
            if res.error: return res

            cases.append((condition, expr))
        if self.current_tok.matches(TT_KEYWORD, 'else'):
            res.register(self.advance())

            else_case = res.register(self.expr())
            if res.error: return res
        if not self.current_tok.matches(TT_KEYWORD, 'end'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'end'"))

        res.register(self.advance())
        return res.success(IfNode(cases, else_case))
            
    def for_expr(self):
        res = ParseResult()
        
        if not self.current_tok.matches(TT_KEYWORD, 'for'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'for'"))

        res.register(self.advance())

        if self.current_tok.tokentype != TT_IDNTIFR:
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected identifier"))

        var_name = self.current_tok
        res.register(self.advance())

        if self.current_tok.tokentype != TT_EQUALS:
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected '='"))
        
        res.register(self.advance())

        start_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.tokentype != TT_COMMA:
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected ','"))

        res.register(self.advance())

        end_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.tokentype == TT_COMMA:
            res.register(self.advance())

            step_value = res.register(self.expr())
            if res.error: return res

        else:
            step_value = None

        if not self.current_tok.matches(TT_KEYWORD, 'do'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'do'"))
        
        res.register(self.advance())

        self.context.symbol_table.set(var_name.tokenValue, Number(start_value))

        body = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'end'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'end'"))
        
        res.register(self.advance())
        
        return res.success(ForNode(var_name, start_value, end_value, step_value,body))

    def while_expr(self):
        res = ParseResult()
        
        if not self.current_tok.matches(TT_KEYWORD, 'while'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'while'"))
        
        res.register(self.advance())

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'do'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'do'"))

        res.register(self.advance())

        body = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TT_KEYWORD, 'end'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'end'"))
        
        res.register(self.advance())
        #self.bin_op(self.comp_expr, ((TT_KEYWORD, "and"), (TT_KEYWORD, "or")))
        #print(self.current_tok)
        return res.success(WhileNode(condition, body))

    def call(self):
        res = ParseResult()
        factor = res.register(self.factor())
        if res.error: return res

        if self.current_tok.tokentype == TT_LPAREN:
            res.register(self.advance())

            arg_nodes = []
            #print(1)
            if self.current_tok.tokentype == TT_RPAREN:
                res.register(self.advance())
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error: return res
                
                while self.current_tok.tokentype == TT_COMMA:
                    res.register(self.advance())

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res

                if self.current_tok.tokentype != TT_RPAREN:
                    return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected ')' or ','"))

                res.register(self.advance())
            return res.success(CallNode(factor,arg_nodes))
        return res.success(factor)

    def factor(self):
        res = ParseResult()
        tok = self.current_tok
        #print(tok.tokentype)
        if tok.tokentype in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        elif tok.tokentype in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.tokentype == TT_IDNTIFR:
            res.register(self.advance())
            return res.success(VarAccessNode(tok))

        elif tok.tokentype == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.tokentype == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', "Expected ')' to close '('"))
        
        elif tok.matches(TT_KEYWORD, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif tok.matches(TT_KEYWORD, 'for'):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif tok.matches(TT_KEYWORD, 'while'):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)
        
        elif tok.matches(TT_KEYWORD, 'func'):
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)

        return res.failure(Error(tok.pos_start, tok.pos_end, 'InvalidSyntaxError', 'Expected INT or FLOAT'))

    def term(self):
        return self.bin_op(self.call, (TT_MUL, TT_DIV, TT_MOD, TT_POW))

    def expr(self):
        res = ParseResult()
        
        if self.current_tok.matches(TT_KEYWORD, 'var'):
            res.register(self.advance())

            if self.current_tok.tokentype != TT_IDNTIFR:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', 'Expected Identifier'))

            if self.current_tok.tokenValue in LOCKED_VARS:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'RunTimeError', f"Couldn't make variable with name '{self.current_tok.tokenValue}'. Please select another name."))

            var_name = self.current_tok
            res.register(self.advance())

            if self.current_tok.tokentype == TT_EQUALS:
                res.register(self.advance())
                expr = res.register(self.expr())
                if res.error: return res
                return res.success(VarAssignNode(var_name, expr))
            #elif self.current_tok.tokentype in (TT_PLUS, TT_MINUS, TT_INT, TT_FLOAT):
                #return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected '=' after {var_name}"))
            else:
                return res.success(VarAssignNode(var_name))

        elif self.current_tok.tokentype == TT_IDNTIFR:
            var = self.context.symbol_table.get(self.current_tok.tokenValue)
            if not var:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'RunTimeError -> Variable not defined', f'{self.current_tok.tokenValue} is not defined'))

            
            var = self.current_tok
            res.register(self.advance())

            if self.current_tok.tokentype == TT_EQUALS:
                if var.tokenValue in LOCKED_VARS:
                    return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'RunTimeError', f"Value of {var.tokenValue} is locked and can't be changed"))
                res.register(self.advance())
                expr = res.register(self.expr())
                if res.error: return res
                return res.success(VarAssignNode(var, expr))
                
            self.current_tok = var
            self.tok_idx -= 1

        return self.bin_op(self.comp_expr, ((TT_KEYWORD, "and"), (TT_KEYWORD, "or")))

    def comp_expr(self):
        res = ParseResult()
        if self.current_tok.matches(TT_KEYWORD, 'not'):
            res.register(self.advance())
            
            
        node = res.register(self.bin_op(self.arith_expr, (TT_ISEQUAL, TT_NOT, TT_SMALLER_THAN, TT_GREATER_THAN, TT_SMALLER_THAN_EQUAL_TO, TT_GREATER_THAN_EQUAL_TO)))
        if res.error:
            if isinstance(res.error, Error):
                return res.failure(res.error)
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', 'Expected int, float, identifier, operator, keywords'))

        return res.success(node)

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def func_def(self):
        res = ParseResult()

        if not self.current_tok.matches(TT_KEYWORD, 'func'):
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected 'func'"))

        res.register(self.advance())

        if self.current_tok.tokentype == TT_IDNTIFR:
            if self.current_tok.tokenValue in LOCKED_VARS:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'RunTimeError', f"invalid name for function")) 
            var_name_tok =  self.current_tok
            res.register(self.advance())
            if not self.current_tok.tokentype == TT_LPAREN:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected '('"))
        else:
            var_name_tok = None
            if not self.current_tok.tokentype == TT_LPAREN:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected '(' or identifier"))
            
        res.register(self.advance())
        arg_name_toks = []

        if self.current_tok.tokentype == TT_IDNTIFR:
            arg_name_toks.append(self.current_tok)
            self.context.symbol_table.set(self.current_tok.tokenValue, self.context.symbol_table.get('null'))
            res.register(self.advance())

            while self.current_tok.tokentype == TT_COMMA:
                res.register(self.advance())

                if self.current_tok.tokentype != TT_IDNTIFR:
                    return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected identifier"))
                
                arg_name_toks.append(self.current_tok)
                self.context.symbol_table.set(self.current_tok.tokenValue, self.context.symbol_table.get('null'))
                res.register(self.advance())
            
            if self.current_tok.tokentype != TT_RPAREN:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected ',' or ')'"))

        else:
            if not self.current_tok.tokentype == TT_RPAREN:
                return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected ')' or identifier"))
        res.register(self.advance())

        if not self.current_tok.tokentype == TT_ARROW:
            return res.failure(Error(self.current_tok.pos_start, self.current_tok.pos_end, 'InvalidSyntaxError', f"Expected '->'"))

        res.register(self.advance())
        node_to_return = res.register(self.expr())
        if res.error: return res
        #print('ayy')
        return res.success(FuncDefNode(var_name_tok, arg_name_toks, node_to_return))



    def bin_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error: return res

        while self.current_tok.tokentype in ops or (self.current_tok.tokentype, self.current_tok.tokenValue) in ops:
            op = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error: return res
            left = BinOpNode(left, op, right)
        return res.success(left)

#######################################
# VALUES
#######################################

class Function:
    def __init__(self, name, body_node, arg_names):
        self.name = name
        self.body_node = body_node
        self.arg_names = arg_names
        self.error = None

        self.set_pos()
        self.set_context()

    def execute(self, args):
        interpreter = Interpreter()
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)

        if len(args) > len(self.arg_names):
            return Error(self.pos_start, self.pos_end, 'RunTimeError', f'Too many arguments passed for function: {self.name}')

        for i in range(len(args)):
            if i == None: i = 0
            arg_name = self.arg_names[i]
            arg_value = args[i]
            arg_value.set_context(new_context)
            new_context.symbol_table.set(arg_name, arg_value)

        value = interpreter.visit(self.body_node, new_context)
        return value


    def set_pos(self, pos_start = None, pos_end = None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"

class Number:
    def __init__(self, value):
        self.value = value
        self.error = None

        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start = None, pos_end = None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context)

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context)

    def multiplied_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context)
    
    def divided_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                result = Number(0)
                result.error = Error(self.pos_start, self.pos_end, 'RunTimeError: Division by 0', f'Attempt to divide {self.value} by 0')
                return result
            return Number(self.value / other.value).set_context(self.context)

    def modded_by(self, other):
        if isinstance(other, Number):
            return Number(self.value % other.value).set_context(self.context)

    def power_of(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context)

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)

class Boolean:
    def __init__(self, value):
        self.error = None
        self.value = value

        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start = None, pos_end = None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def comp_equal(self, other):
        if isinstance(other, Number):
            return Boolean(self.value.value == other.value and 'true' or 'false').set_context(self.context)
        else:
            return other

    def comp_greater(self, other):
        if isinstance(other, Number):
            return Boolean((self.value.value == 'true' and 1 or self.value.value == 'false' and 0) > other.value and 'true' or 'false').set_context(self.context)
        else:
            return other

    def comp_lesser(self, other):
        if isinstance(other, Number):
            return Boolean((self.value.value == 'true' and 1 or self.value.value == 'false' and 0) < other.value and 'true' or 'false').set_context(self.context)
        else:
            return other
    def comp_not_equal(self, other):
        if isinstance(other, Number):
            return Boolean(self.value.value != other.value and 'true' or 'false').set_context(self.context)
        else:
            return other

    def comp_greater_equal(self, other):
        if isinstance(other, Number):
            return Boolean(self.value.value >= other.value and 'true' or 'false').set_context(self.context)
        else:
            return other

    def comp_lesser_equal(self, other):
        if isinstance(other, Number):
            return Boolean(self.value.value <= other.value and 'true' or 'false').set_context(self.context)
        else:
            return other

    def comp_greater_equal(self, other):
        if isinstance(other, Number):
            return Boolean(self.value.value == other.value and 'true' or 'false').set_context(self.context)
        else:
            return other

    def comp_and(self, other):
        if isinstance(other, Number):
            return Boolean(self.value.value and other.value).set_context(self.context)
        else:
            if self.value.value=='true' and other.value=='true':
                return Boolean('true').set_context(self.context)
            else:
                return Boolean('false').set_context(self.context)
    
    def comp_or(self, other):
        if isinstance(other, Number):
            return Boolean(self.value.value or other.value).set_context(self.context)
        else:
            if self.value.value=='true' or other.value=='true':
                return Boolean('true').set_context(self.context)
            else:
                return Boolean('false').set_context(self.context)

    def added_to(self, other):
        if isinstance(other, Number):
            if self.value == 'false':
                return other
            elif self.value == 'true':
                return Number(other.value + 1)

    def subbed_by(self, other):
        if isinstance(other, Number):
            if self.value == 'false':
                return other
            elif self.value == 'true':
                return Number(other.value + 1)

    def multiplied_by(self, other):
        if isinstance(other, Number):
            if self.value == 'false':
                return other
            elif self.value == 'true':
                return Number(other.value + 1)

    def divided_by(self, other):
        if isinstance(other, Number):
            if self.value == 'false':
                return other
            elif self.value == 'true':
                return Number(other.value + 1)

    def modded_by(self, other):
        if isinstance(other, Number):
            if self.value == 'false':
                return other
            elif self.value == 'true':
                return Number(other.value + 1)

    def power_of(self, other):
        if isinstance(other, Number):
            if self.value == 'false':
                return other
            elif self.value == 'true':
                return Number(other.value + 1)
            

    def copy(self):
        copy = Boolean(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)

#######################################
# CONTEXT
#######################################

class Context:
    def __init__(self, display_name, parent = None, parent_entry_pos = None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

#######################################
# SYMBOL TABLE
#######################################

class SymbolTable:
    def __init__(self, parent = None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]

#######################################
# INTERPRETER
#######################################

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    #######################################

    def visit_NumberNode(self, node, context):
        return Number(node.tok.tokenValue).set_pos(node.pos_start, node.pos_end).set_context(context)
    
    def visit_BinOpNode(self, node, context):
        left = self.visit(node.left_node, context)
        right = self.visit(node.right_node, context)

        if isinstance(left, Error):
            return left
        
        if isinstance(right, Error):
            return right

        if left.error: return left
        if right.error: return right
        #print(left, right)

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
        elif node.op.tokentype == TT_ISEQUAL:
            left = Boolean(left)
            result = left.comp_equal(right)
        elif node.op.tokentype == TT_SMALLER_THAN_EQUAL_TO:
            left = Boolean(left)
            result = left.comp_lesser_equal(right)
        elif node.op.tokentype == TT_GREATER_THAN_EQUAL_TO:
            left = Boolean(left)
            result = left.comp_greater_equal(right)
        elif node.op.tokentype == TT_SMALLER_THAN:
            left = Boolean(left)
            result = left.comp_lesser(right)
        elif node.op.tokentype == TT_GREATER_THAN:
            left = Boolean(left)
            result = left.comp_greater(right)
        elif node.op.tokentype == TT_NOT:
            left = Boolean(left)
            result = left.comp_not_equal(right)
        elif node.op.tokentype == TT_KEYWORD and node.op.tokenValue == 'and':
            left = Boolean(left)
            result = left.comp_and(right)
        elif node.op.tokentype == TT_KEYWORD and node.op.tokenValue == 'or':
            left = Boolean(left)
            result = left.comp_or(right)


        if result.error: return result
        return result.set_pos(node.pos_start, node.pos_end)

    def visit_VarAccessNode(self, node, context):
        var_name = node.var_name_token.tokenValue if isinstance(node.var_name_token, Token) else node.var_name_token
        value = context.symbol_table.get(var_name)

        if not value:
            value = Number(0)
            return Error(node.pos_start, node.pos_end, 'RunTimeError -> Variable not defined', f'{var_name} is not defined')

        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return value

    def visit_VarAssignNode(self, node, context):
        var_name = node.var_name_token.tokenValue
        value = None
        if node.valueNode: value = self.visit(node.valueNode, context)
        
        context.symbol_table.set(var_name, value or context.symbol_table.get('null'))
        return value


    def visit_UnaryOpNode(self, node, context):
        number = self.visit(node.node, context)
        
        if node.op.tokentype == TT_MINUS:
            number = number.multiplied_by(Number(-1))

        return number.set_pos(node.pos_start, node.pos_end)

    def visit_IfNode(self, node, context):
        for condition, expr in node.cases:
            condition_value = self.visit(condition, context)
            #print('CONDITION VALUE:',condition_value.value)

            if condition_value.value !='false' and str(condition_value.value) != '0':
                expr_value = self.visit(expr, context)
                return expr_value
        if node.else_case:
            else_value = self.visit(node.else_case, context)
            return else_value
        return None

    def visit_ForNode(self, node, context):
        start_value = self.visit(node.start_value_node, context)
        end_value = self.visit(node.end_value_node, context)
        if node.increment:
            step_value = self.visit(node.increment, context)
        else:
            step_value = Number(1)

        i = start_value.value
        if i=='true': i = 1
        elif i == 'false': i = 0

        if step_value.value == 'true':
            step_value.value = 1
        elif step_value.value == 'false':
            step_value.value = 0

        if end_value.value == 'true':
            end_value.value = 1
        elif end_value.value == 'false':
            end_value.value = 0

        if step_value.value >= 0:
            condition = lambda: i <= end_value.value
        else:
            condition = lambda: i >= end_value.value

        while condition():
            context.symbol_table.set(node.var_name_tok.tokenValue, Number(i))
            i += step_value.value

            res = self.visit(node.body_node, context)
            #if res: print(res.value)

        return None
    
    def visit_WhileNode(self, node, context):
        while True:
            condition = self.visit(node.condition_node, context)

            if not condition.value == 'true': break
            res = self.visit(node.body_node, context)
            #print(res.value)

        return None

    def visit_FuncDefNode(self, node, context):
        func_name = node.var_name_tok.tokenValue if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.tokenValue for arg_name in node.arg_name_tokens]
        func_value = Function(func_name, body_node, arg_names).set_context(context).set_pos(node.pos_start, node.pos_end)
        #print(func_name, body_node, arg_names, func_value)
        if node.var_name_tok:
            context.symbol_table.set(func_name, func_value)

        #print(context.symbol_table.symbols)
        return func_value

    def visit_CallNode(self, node, context):
        args = []

        value_to_call = self.visit(node.node_to_call, context)
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)
        for arg_node in node.arg_nodes:
            args.append(self.visit(arg_node, context))

        return_value = value_to_call.execute(args) if isinstance(value_to_call, Function) else Error(node.pos_start, node.pos_end, 'RunTimeError', f'Expected valid function')
        
        if value_to_call.name == 'print':
            print(return_value)
            return_value = None
        elif value_to_call.name == 'type':
            print(type(return_value))

        if value_to_call.name in LOCKED_VARS: return_value = None

        return return_value



global_symbol_table = SymbolTable()
global_symbol_table.set('null', Number(0))
global_symbol_table.set('true', Boolean('true'))
global_symbol_table.set('false', Boolean('false'))

global_symbol_table.set('print', Function('print', VarAccessNode('a'), ['a']).set_pos().set_context(Context('<module>')))
global_symbol_table.set('type', Function('type', VarAccessNode('a'), ['a']).set_pos().set_context(Context('<module>')))

#######################################
# RUN
#######################################
def run(fn, code):
    lexer = Lexer(fn, code)
    tokens, error = lexer.Make_Tokens()
    if error: return None, error
    #print(tokens)

    # Create context table
    context = Context('<module>')
    context.symbol_table = global_symbol_table

    # Generate AST
    if len(tokens) <= 1: return [], None
    parser = Parser(tokens, context)
    ast = parser.parse()
    if ast.error: return None, ast.error
    #print(ast.node)

    # Run program
    interpreter = Interpreter()
    result = interpreter.visit(ast.node, context)
    if isinstance(result, Error):
        return None, result
    if result == None: return None, None
    if result.error: return None, result.error

    return result, None
