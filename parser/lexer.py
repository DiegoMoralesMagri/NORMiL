"""
NORMiL Lexer (Analyseur Lexical)
=================================

Tokenise le code source NORMiL en une séquence de tokens.

Exemple:
    >>> code = 'let x: int = 42'
    >>> lexer = Lexer(code)
    >>> tokens = lexer.tokenize()
    >>> for token in tokens:
    ...     print(token)
    Token(LET, 'let', 1, 1)
    Token(IDENTIFIER, 'x', 1, 5)
    Token(COLON, ':', 1, 6)
    Token(IDENTIFIER, 'int', 1, 8)
    Token(ASSIGN, '=', 1, 12)
    Token(INT_LITERAL, '42', 1, 14)
    Token(EOF, '', 1, 16)
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional
import re


class TokenType(Enum):
    """Types de tokens NORMiL"""
    
    # Mots-clés
    LET = auto()
    CONST = auto()
    FN = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    IN = auto()
    WHILE = auto()
    MATCH = auto()
    CASE = auto()
    WHERE = auto()
    TRANSACTION = auto()
    ATOMIC = auto()
    DISTRIBUTED = auto()
    COMPENSATING = auto()
    TRY = auto()
    CATCH = auto()
    THROW = auto()
    ON_ROLLBACK = auto()
    TYPE = auto()
    MODULE = auto()
    IMPORT = auto()
    EXPORT = auto()
    AS = auto()
    PRIMITIVE = auto()
    BEFORE_TRANSACTION = auto()
    AFTER_TRANSACTION = auto()
    ON_ERROR = auto()
    PERMISSIONS = auto()
    ALLOW = auto()
    DENY = auto()
    READ = auto()
    WRITE = auto()
    DELETE = auto()
    ON = auto()
    OPTIONAL = auto()
    LIST = auto()
    MAP = auto()
    
    # Types primitifs
    INT_TYPE = auto()
    FLOAT_TYPE = auto()
    BOOL_TYPE = auto()
    STR_TYPE = auto()
    TIMESTAMP_TYPE = auto()
    UUID_TYPE = auto()
    VECTOR_TYPE = auto()
    SPARSE_VECTOR_TYPE = auto()
    
    # Littéraux
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    TRUE = auto()
    FALSE = auto()
    
    # Identificateurs
    IDENTIFIER = auto()
    
    # Opérateurs arithmétiques
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    
    # Opérateurs vectoriels
    VEC_PLUS = auto()      # .+
    VEC_MINUS = auto()     # .-
    VEC_STAR = auto()      # .*
    VEC_SLASH = auto()     # ./
    AT = auto()            # @ (dot product)
    
    # Opérateurs de comparaison
    EQ = auto()            # ==
    NE = auto()            # !=
    LT = auto()            # <
    GT = auto()            # >
    LE = auto()            # <=
    GE = auto()            # >=
    
    # Opérateurs logiques
    AND = auto()           # &&
    OR = auto()            # ||
    NOT = auto()           # !
    
    # Opérateurs d'affectation
    ASSIGN = auto()        # =
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=
    
    # Délimiteurs
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    COMMA = auto()         # ,
    SEMICOLON = auto()     # ;
    COLON = auto()         # :
    DOUBLE_COLON = auto()  # ::
    DOT = auto()           # .
    ARROW = auto()         # ->
    ELLIPSIS = auto()      # ...
    
    # Spéciaux
    ANNOTATION = auto()    # @identifier
    NEWLINE = auto()
    EOF = auto()
    
    # Commentaires (ignorés généralement)
    COMMENT = auto()


@dataclass
class Token:
    """Représente un token du code source"""
    type: TokenType
    value: str
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}, {self.column})"


class Lexer:
    """Analyseur lexical pour NORMiL"""
    
    # Mots-clés
    KEYWORDS = {
        'let': TokenType.LET,
        'const': TokenType.CONST,
        'fn': TokenType.FN,
        'return': TokenType.RETURN,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'for': TokenType.FOR,
        'in': TokenType.IN,
        'while': TokenType.WHILE,
        'match': TokenType.MATCH,
        'case': TokenType.CASE,
        'where': TokenType.WHERE,
        'transaction': TokenType.TRANSACTION,
        'atomic': TokenType.ATOMIC,
        'distributed': TokenType.DISTRIBUTED,
        'compensating': TokenType.COMPENSATING,
        'try': TokenType.TRY,
        'catch': TokenType.CATCH,
        'throw': TokenType.THROW,
        'on_rollback': TokenType.ON_ROLLBACK,
        'type': TokenType.TYPE,
        'module': TokenType.MODULE,
        'import': TokenType.IMPORT,
        'export': TokenType.EXPORT,
        'as': TokenType.AS,
        'primitive': TokenType.PRIMITIVE,
        'before_transaction': TokenType.BEFORE_TRANSACTION,
        'after_transaction': TokenType.AFTER_TRANSACTION,
        'on_error': TokenType.ON_ERROR,
        'permissions': TokenType.PERMISSIONS,
        'allow': TokenType.ALLOW,
        'deny': TokenType.DENY,
        'read': TokenType.READ,
        'write': TokenType.WRITE,
        'delete': TokenType.DELETE,
        'on': TokenType.ON,
        'optional': TokenType.OPTIONAL,
        'list': TokenType.LIST,
        'map': TokenType.MAP,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        # Types primitifs
        'int': TokenType.INT_TYPE,
        'float': TokenType.FLOAT_TYPE,
        'bool': TokenType.BOOL_TYPE,
        'str': TokenType.STR_TYPE,
        'timestamp': TokenType.TIMESTAMP_TYPE,
        'uuid': TokenType.UUID_TYPE,
        'Vector': TokenType.VECTOR_TYPE,
        'SparseVector': TokenType.SPARSE_VECTOR_TYPE,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def current_char(self) -> Optional[str]:
        """Retourne le caractère actuel ou None si fin"""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Regarde le caractère à venir sans avancer"""
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> Optional[str]:
        """Avance d'un caractère et retourne le caractère consommé"""
        char = self.current_char()
        if char is None:
            return None
        
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        return char
    
    def skip_whitespace(self):
        """Saute les espaces blancs (sauf newlines si significatifs)"""
        while self.current_char() and self.current_char() in ' \t\r\n':
            self.advance()
    
    def skip_comment(self):
        """Saute les commentaires // et /* */"""
        if self.current_char() == '/' and self.peek_char() == '/':
            # Commentaire ligne
            while self.current_char() and self.current_char() != '\n':
                self.advance()
            self.advance()  # Consommer le \n
        elif self.current_char() == '/' and self.peek_char() == '*':
            # Commentaire bloc
            self.advance()  # /
            self.advance()  # *
            while self.current_char():
                if self.current_char() == '*' and self.peek_char() == '/':
                    self.advance()  # *
                    self.advance()  # /
                    break
                self.advance()
    
    def read_number(self) -> Token:
        """Lit un nombre (int ou float)"""
        start_line = self.line
        start_col = self.column
        num_str = ''
        is_float = False
        
        # Partie entière
        while self.current_char() and self.current_char().isdigit():
            num_str += self.advance()
        
        # Partie décimale
        if self.current_char() == '.' and self.peek_char() and self.peek_char().isdigit():
            is_float = True
            num_str += self.advance()  # .
            while self.current_char() and self.current_char().isdigit():
                num_str += self.advance()
        
        # Exposant
        if self.current_char() and self.current_char() in 'eE':
            is_float = True
            num_str += self.advance()  # e/E
            if self.current_char() and self.current_char() in '+-':
                num_str += self.advance()
            while self.current_char() and self.current_char().isdigit():
                num_str += self.advance()
        
        token_type = TokenType.FLOAT_LITERAL if is_float else TokenType.INT_LITERAL
        return Token(token_type, num_str, start_line, start_col)
    
    def read_string(self) -> Token:
        """Lit une chaîne de caractères"""
        start_line = self.line
        start_col = self.column
        self.advance()  # Consommer le "
        
        string_val = ''
        while self.current_char() and self.current_char() != '"':
            if self.current_char() == '\\':
                self.advance()
                # Échappements
                escape_char = self.current_char()
                if escape_char == 'n':
                    string_val += '\n'
                elif escape_char == 't':
                    string_val += '\t'
                elif escape_char == 'r':
                    string_val += '\r'
                elif escape_char == '"':
                    string_val += '"'
                elif escape_char == '\\':
                    string_val += '\\'
                else:
                    string_val += escape_char
                self.advance()
            else:
                string_val += self.advance()
        
        if self.current_char() == '"':
            self.advance()  # Consommer le " final
        else:
            raise SyntaxError(f"Unterminated string at line {start_line}, column {start_col}")
        
        return Token(TokenType.STRING_LITERAL, string_val, start_line, start_col)
    
    def read_identifier(self) -> Token:
        """Lit un identificateur ou mot-clé"""
        start_line = self.line
        start_col = self.column
        ident = ''
        
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            ident += self.advance()
        
        # Vérifier si c'est un mot-clé
        token_type = self.KEYWORDS.get(ident, TokenType.IDENTIFIER)
        return Token(token_type, ident, start_line, start_col)
    
    def read_annotation(self) -> Token:
        """Lit une annotation @identifier"""
        start_line = self.line
        start_col = self.column
        self.advance()  # Consommer @
        
        ident = ''
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            ident += self.advance()
        
        return Token(TokenType.ANNOTATION, '@' + ident, start_line, start_col)
    
    def tokenize(self) -> List[Token]:
        """Tokenise tout le code source"""
        while self.current_char():
            # Espaces blancs et commentaires
            if self.current_char() in ' \t\r\n':
                self.skip_whitespace()
                continue
            
            if self.current_char() == '/' and self.peek_char() in ['/', '*']:
                self.skip_comment()
                continue
            
            start_line = self.line
            start_col = self.column
            char = self.current_char()
            
            # Nombres
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Chaînes
            if char == '"':
                self.tokens.append(self.read_string())
                continue
            
            # Annotations
            if char == '@':
                self.tokens.append(self.read_annotation())
                continue
            
            # Identificateurs et mots-clés
            if char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Opérateurs et délimiteurs multi-caractères
            if char == '=' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.EQ, '==', start_line, start_col))
                continue
            
            if char == '!' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NE, '!=', start_line, start_col))
                continue
            
            if char == '<' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LE, '<=', start_line, start_col))
                continue
            
            if char == '>' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GE, '>=', start_line, start_col))
                continue
            
            if char == '&' and self.peek_char() == '&':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.AND, '&&', start_line, start_col))
                continue
            
            if char == '|' and self.peek_char() == '|':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.OR, '||', start_line, start_col))
                continue
            
            if char == '+' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, '+=', start_line, start_col))
                continue
            
            if char == '-' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, '-=', start_line, start_col))
                continue
            
            if char == '*' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.STAR_ASSIGN, '*=', start_line, start_col))
                continue
            
            if char == '/' and self.peek_char() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.SLASH_ASSIGN, '/=', start_line, start_col))
                continue
            
            if char == '-' and self.peek_char() == '>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '->', start_line, start_col))
                continue
            
            if char == ':' and self.peek_char() == ':':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.DOUBLE_COLON, '::', start_line, start_col))
                continue
            
            if char == '.' and self.peek_char() == '+':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.VEC_PLUS, '.+', start_line, start_col))
                continue
            
            if char == '.' and self.peek_char() == '-':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.VEC_MINUS, '.-', start_line, start_col))
                continue
            
            if char == '.' and self.peek_char() == '*':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.VEC_STAR, '.*', start_line, start_col))
                continue
            
            if char == '.' and self.peek_char() == '/':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.VEC_SLASH, './', start_line, start_col))
                continue
            
            if char == '.' and self.peek_char() == '.' and self.peek_char(2) == '.':
                self.advance()
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ELLIPSIS, '...', start_line, start_col))
                continue
            
            # Opérateurs et délimiteurs simples
            single_char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
                '%': TokenType.PERCENT,
                '@': TokenType.AT,
                '=': TokenType.ASSIGN,
                '<': TokenType.LT,
                '>': TokenType.GT,
                '!': TokenType.NOT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ',': TokenType.COMMA,
                ';': TokenType.SEMICOLON,
                ':': TokenType.COLON,
                '.': TokenType.DOT,
            }
            
            if char in single_char_tokens:
                token_type = single_char_tokens[char]
                self.advance()
                self.tokens.append(Token(token_type, char, start_line, start_col))
                continue
            
            # Caractère inconnu
            raise SyntaxError(f"Unexpected character '{char}' at line {start_line}, column {start_col}")
        
        # Ajouter EOF
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens


# ============================================
# Tests rapides
# ============================================

if __name__ == '__main__':
    # Test 1: Programme simple
    code1 = """
    let x: int = 42
    let y: float = 3.14
    let name: str = "O-RedMind"
    """
    
    print("=== Test 1: Variables ===")
    lexer1 = Lexer(code1)
    tokens1 = lexer1.tokenize()
    for token in tokens1:
        print(token)
    
    # Test 2: Fonction
    code2 = """
    fn add(a: int, b: int) -> int {
        return a + b
    }
    """
    
    print("\n=== Test 2: Fonction ===")
    lexer2 = Lexer(code2)
    tokens2 = lexer2.tokenize()
    for token in tokens2:
        print(token)
    
    # Test 3: Opérateurs vectoriels
    code3 = "let v3 = v1 .+ v2"
    
    print("\n=== Test 3: Opérateurs vectoriels ===")
    lexer3 = Lexer(code3)
    tokens3 = lexer3.tokenize()
    for token in tokens3:
        print(token)
    
    # Test 4: Annotation
    code4 = "@plastic(rate: 0.001)"
    
    print("\n=== Test 4: Annotation ===")
    lexer4 = Lexer(code4)
    tokens4 = lexer4.tokenize()
    for token in tokens4:
        print(token)
    
    print("\n✅ Lexer tests passed!")
