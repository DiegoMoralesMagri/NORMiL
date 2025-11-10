"""
NORMiL Parser (Analyseur Syntaxique)
=====================================

Auteur : Diego Morales Magri

Parse les tokens en AST (Abstract Syntax Tree).
Utilise un parseur récursif descendant.

Exemple:
    >>> from parser.lexer import Lexer
    >>> from parser.parser import Parser
    >>> 
    >>> code = 'let x: int = 42'
    >>> lexer = Lexer(code)
    >>> tokens = lexer.tokenize()
    >>> parser = Parser(tokens)
    >>> ast = parser.parse()
    >>> print(ast)
"""

from typing import List, Optional, Dict
from .lexer import Token, TokenType
from .ast_nodes import *


class ParseError(Exception):
    """Erreur de parsing"""
    pass


class Parser:
    """Parseur récursif descendant pour NORMiL"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    # ============================================
    # Helpers
    # ============================================
    
    def current_token(self) -> Token:
        """Retourne le token actuel"""
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[self.pos]
    
    def peek_token(self, offset: int = 1) -> Token:
        """Regarde le token à venir"""
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[pos]
    
    def advance(self) -> Token:
        """Avance d'un token et retourne le token consommé"""
        token = self.current_token()
        if token.type != TokenType.EOF:
            self.pos += 1
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        """Consomme un token du type attendu ou lève une erreur"""
        token = self.current_token()
        if token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {token.type.name} "
                f"at line {token.line}, column {token.column}"
            )
        return self.advance()
    
    def match(self, *token_types: TokenType) -> bool:
        """Vérifie si le token actuel correspond à un des types"""
        return self.current_token().type in token_types
    
    def consume(self, token_type: TokenType) -> bool:
        """Consomme le token s'il correspond, sinon retourne False"""
        if self.match(token_type):
            self.advance()
            return True
        return False
    
    # ============================================
    # Types
    # ============================================
    
    def parse_type(self) -> Type:
        """Parse un type"""
        token = self.current_token()
        
        # Types primitifs
        if token.type in (TokenType.INT_TYPE, TokenType.FLOAT_TYPE, 
                         TokenType.BOOL_TYPE, TokenType.STR_TYPE,
                         TokenType.TIMESTAMP_TYPE, TokenType.UUID_TYPE):
            type_name = token.value
            self.advance()
            return PrimitiveType(type_name)
        
        # Vector<float, dim=256, q=8>
        if token.type == TokenType.VECTOR_TYPE:
            self.advance()
            self.expect(TokenType.LT)
            
            # Type élément
            elem_type = self.parse_type()
            self.expect(TokenType.COMMA)
            
            # dim=...
            self.expect(TokenType.IDENTIFIER)  # "dim"
            self.expect(TokenType.ASSIGN)
            dim_token = self.expect(TokenType.INT_LITERAL)
            dim = int(dim_token.value)
            
            # q=... (optionnel)
            quantization = None
            if self.consume(TokenType.COMMA):
                self.expect(TokenType.IDENTIFIER)  # "q"
                self.expect(TokenType.ASSIGN)
                q_token = self.expect(TokenType.INT_LITERAL)
                quantization = int(q_token.value)
            
            self.expect(TokenType.GT)
            return VectorType(elem_type, dim, quantization)
        
        # list<Type>
        if token.type == TokenType.LIST:
            self.advance()
            self.expect(TokenType.LT)
            elem_type = self.parse_type()
            self.expect(TokenType.GT)
            return ListType(elem_type)
        
        # map<KeyType, ValueType>
        if token.type == TokenType.MAP:
            self.advance()
            self.expect(TokenType.LT)
            key_type = self.parse_type()
            self.expect(TokenType.COMMA)
            value_type = self.parse_type()
            self.expect(TokenType.GT)
            return MapType(key_type, value_type)
        
        # optional<Type>
        if token.type == TokenType.OPTIONAL:
            self.advance()
            self.expect(TokenType.LT)
            inner_type = self.parse_type()
            self.expect(TokenType.GT)
            return OptionalType(inner_type)
        
        # Type nommé (défini par utilisateur)
        if token.type == TokenType.IDENTIFIER:
            type_name = token.value
            self.advance()
            return NamedType(type_name)
        
        # Struct type inline: {field1: Type1, field2: Type2}
        if token.type == TokenType.LBRACE:
            self.advance()
            fields = {}
            
            while not self.match(TokenType.RBRACE):
                field_name = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.COLON)
                field_type = self.parse_type()
                fields[field_name] = field_type
                
                if not self.consume(TokenType.COMMA):
                    break
            
            self.expect(TokenType.RBRACE)
            return StructType(fields)
        
        raise ParseError(
            f"Expected type, got {token.type.name} at line {token.line}, column {token.column}"
        )
    
    # ============================================
    # Expressions
    # ============================================
    
    def parse_primary_expression(self) -> Expression:
        """Parse une expression primaire"""
        token = self.current_token()
        
        # Littéraux
        if token.type == TokenType.INT_LITERAL:
            value = int(token.value)
            self.advance()
            return IntLiteral(value)
        
        if token.type == TokenType.FLOAT_LITERAL:
            value = float(token.value)
            self.advance()
            return FloatLiteral(value)
        
        if token.type == TokenType.STRING_LITERAL:
            value = token.value
            self.advance()
            return StringLiteral(value)
        
        if token.type == TokenType.TRUE:
            self.advance()
            return BoolLiteral(True)
        
        if token.type == TokenType.FALSE:
            self.advance()
            return BoolLiteral(False)
        
        # Identificateur ou Struct nommé (TypeName { fields })
        if token.type == TokenType.IDENTIFIER:
            name = token.value
            self.advance()
            
            # Vérifier si c'est un struct nommé: TypeName { ... }
            # IMPORTANT: Seulement si le nom commence par une MAJUSCULE (convention de nommage)
            # Ceci évite de confondre `match n {` avec un struct literal
            if self.match(TokenType.LBRACE) and name[0].isupper():
                self.advance()
                
                # Parser les champs du struct
                struct_fields = {}
                
                while not self.match(TokenType.RBRACE):
                    # Nom du champ - accepter identifier ou mots-clés avec valeur
                    current = self.current_token()
                    
                    # Tout token avec une valeur peut être un nom de champ
                    if hasattr(current, 'value') and current.value:
                        field_name = current.value
                        self.advance()
                    else:
                        raise ParseError(f"Expected field name in struct literal, got {current.type.name} at line {current.line}")
                    
                    self.expect(TokenType.COLON)
                    
                    # Valeur du champ
                    field_value = self.parse_expression()
                    struct_fields[field_name] = field_value
                    
                    # Virgule optionnelle entre les champs
                    if not self.consume(TokenType.COMMA):
                        break
                
                self.expect(TokenType.RBRACE)
                return StructLiteral(name, struct_fields)
            
            # Sinon, c'est juste un identificateur
            return Identifier(name)
        
        # Expression entre parenthèses
        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        # Liste: [elem1, elem2, ...]
        if token.type == TokenType.LBRACKET:
            self.advance()
            elements = []
            
            while not self.match(TokenType.RBRACKET):
                elements.append(self.parse_expression())
                if not self.consume(TokenType.COMMA):
                    break
            
            self.expect(TokenType.RBRACKET)
            return ListLiteral(elements)
        
        # Map ou Struct: {key: val, ...}
        if token.type == TokenType.LBRACE:
            self.advance()
            
            # Map vide ou struct vide
            if self.match(TokenType.RBRACE):
                self.advance()
                return StructLiteral(None, {})  # Struct vide anonyme
            
            # Vérifier si c'est un struct (clés = identifiers) ou map (clés = expressions)
            is_struct = False
            struct_fields = {}
            map_entries = []
            
            # Première entrée
            first_token = self.current_token()
            if first_token.type == TokenType.IDENTIFIER:
                # Pourrait être un struct
                field_name = first_token.value
                self.advance()
                if self.match(TokenType.COLON):
                    # C'est un struct!
                    is_struct = True
                    self.advance()
                    field_value = self.parse_expression()
                    struct_fields[field_name] = field_value
                else:
                    # Oups, c'était une expression complexe, retour arrière nécessaire
                    # Pour l'instant, erreur simple
                    raise ParseError(f"Expected ':' after field name '{field_name}'")
            else:
                # Map: première clé est une expression
                first_key = self.parse_expression()
                self.expect(TokenType.COLON)
                first_value = self.parse_expression()
                map_entries.append((first_key, first_value))
            
            # Entrées suivantes
            while self.consume(TokenType.COMMA):
                if self.match(TokenType.RBRACE):
                    break
                
                if is_struct:
                    # Parser comme champ de struct
                    if not self.match(TokenType.IDENTIFIER):
                        raise ParseError(f"Expected field name in struct literal")
                    field_name = self.current_token().value
                    self.advance()
                    self.expect(TokenType.COLON)
                    field_value = self.parse_expression()
                    struct_fields[field_name] = field_value
                else:
                    # Parser comme entrée de map
                    key = self.parse_expression()
                    self.expect(TokenType.COLON)
                    value = self.parse_expression()
                    map_entries.append((key, value))
            
            self.expect(TokenType.RBRACE)
            
            if is_struct:
                return StructLiteral(None, struct_fields)
            else:
                return MapLiteral(map_entries)
        
        raise ParseError(
            f"Expected expression, got {token.type.name} at line {token.line}, column {token.column}"
        )
    
    def parse_postfix_expression(self) -> Expression:
        """Parse expression avec postfix (call, field, index)"""
        expr = self.parse_primary_expression()
        
        while True:
            token = self.current_token()
            
            # Function call: expr(args)
            if token.type == TokenType.LPAREN:
                self.advance()
                arguments = []
                
                while not self.match(TokenType.RPAREN):
                    # Vérifier si c'est un argument nommé (name: value)
                    # Lookahead: si IDENTIFIER suivi de COLON
                    if (self.current_token().type == TokenType.IDENTIFIER and 
                        self.peek_token(1).type == TokenType.COLON):
                        # Argument nommé
                        name = self.expect(TokenType.IDENTIFIER).value
                        self.expect(TokenType.COLON)
                        value = self.parse_expression()
                        from parser.ast_nodes import NamedArgument
                        arguments.append(NamedArgument(name, value))
                    else:
                        # Argument positionnel
                        arguments.append(self.parse_expression())
                    
                    if not self.consume(TokenType.COMMA):
                        break
                
                self.expect(TokenType.RPAREN)
                expr = FunctionCall(expr, arguments)
            
            # Field access: expr.field
            elif token.type == TokenType.DOT:
                self.advance()
                # Accepter identifier ou mot-clé comme nom de champ
                current = self.current_token()
                if hasattr(current, 'value') and current.value:
                    field_name = current.value
                    self.advance()
                else:
                    raise ParseError(f"Expected field name after '.', got {current.type.name} at line {current.line}")
                expr = FieldAccess(expr, field_name)
            
            # Index access: expr[index]
            elif token.type == TokenType.LBRACKET:
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexAccess(expr, index)
            
            else:
                break
        
        return expr
    
    def parse_unary_expression(self) -> Expression:
        """Parse expression unaire"""
        token = self.current_token()
        
        if token.type in (TokenType.MINUS, TokenType.NOT):
            op = token.value
            self.advance()
            operand = self.parse_unary_expression()
            return UnaryOp(op, operand)
        
        return self.parse_postfix_expression()
    
    def parse_multiplicative_expression(self) -> Expression:
        """Parse expression multiplicative: *, /, %, .*, ./, @"""
        left = self.parse_unary_expression()
        
        while self.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT,
                         TokenType.VEC_STAR, TokenType.VEC_SLASH, TokenType.AT):
            op = self.current_token().value
            self.advance()
            right = self.parse_unary_expression()
            left = BinaryOp(op, left, right)
        
        return left
    
    def parse_additive_expression(self) -> Expression:
        """Parse expression additive: +, -, .+, .-"""
        left = self.parse_multiplicative_expression()
        
        while self.match(TokenType.PLUS, TokenType.MINUS,
                         TokenType.VEC_PLUS, TokenType.VEC_MINUS):
            op = self.current_token().value
            self.advance()
            right = self.parse_multiplicative_expression()
            left = BinaryOp(op, left, right)
        
        return left
    
    def parse_relational_expression(self) -> Expression:
        """Parse expression relationnelle: <, >, <=, >="""
        left = self.parse_additive_expression()
        
        while self.match(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op = self.current_token().value
            self.advance()
            right = self.parse_additive_expression()
            left = BinaryOp(op, left, right)
        
        return left
    
    def parse_equality_expression(self) -> Expression:
        """Parse expression d'égalité: ==, !="""
        left = self.parse_relational_expression()
        
        while self.match(TokenType.EQ, TokenType.NE):
            op = self.current_token().value
            self.advance()
            right = self.parse_relational_expression()
            left = BinaryOp(op, left, right)
        
        return left
    
    def parse_logical_and_expression(self) -> Expression:
        """Parse expression AND: &&"""
        left = self.parse_equality_expression()
        
        while self.match(TokenType.AND):
            op = self.current_token().value
            self.advance()
            right = self.parse_equality_expression()
            left = BinaryOp(op, left, right)
        
        return left
    
    def parse_logical_or_expression(self) -> Expression:
        """Parse expression OR: ||"""
        left = self.parse_logical_and_expression()
        
        while self.match(TokenType.OR):
            op = self.current_token().value
            self.advance()
            right = self.parse_logical_and_expression()
            left = BinaryOp(op, left, right)
        
        return left
    
    def parse_assignment_expression(self) -> Expression:
        """Parse expression d'affectation: =, +=, -=, etc."""
        left = self.parse_logical_or_expression()
        
        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN,
                     TokenType.MINUS_ASSIGN, TokenType.STAR_ASSIGN,
                     TokenType.SLASH_ASSIGN):
            op = self.current_token().value
            self.advance()
            right = self.parse_assignment_expression()
            left = BinaryOp(op, left, right)
        
        return left
    
    def parse_expression(self) -> Expression:
        """Parse une expression complète"""
        return self.parse_assignment_expression()
    
    # ============================================
    # Statements
    # ============================================
    
    def parse_block(self) -> Block:
        """Parse un bloc: { stmt1; stmt2; }"""
        self.expect(TokenType.LBRACE)
        statements = []
        
        while not self.match(TokenType.RBRACE):
            statements.append(self.parse_statement())
        
        self.expect(TokenType.RBRACE)
        return Block(statements)
    
    def parse_var_decl(self) -> VarDecl:
        """Parse: let x: int = 42 ou let x = 42 (inférence) ou const x = 42"""
        is_const = self.consume(TokenType.CONST)
        if not is_const:
            self.expect(TokenType.LET)
        
        name = self.expect(TokenType.IDENTIFIER).value
        
        # Type optionnel (si absent, inférence automatique)
        var_type = None
        if self.consume(TokenType.COLON):
            var_type = self.parse_type()
        
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()
        
        return VarDecl(name, var_type, value, is_const)
    
    def parse_function_decl(self, annotations: List[Annotation] = None) -> FunctionDecl:
        """Parse: fn name(params) -> Type { body }"""
        self.expect(TokenType.FN)
        name = self.expect(TokenType.IDENTIFIER).value
        
        # Paramètres
        self.expect(TokenType.LPAREN)
        parameters = []
        
        while not self.match(TokenType.RPAREN):
            param_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            param_type = self.parse_type()
            
            # Valeur par défaut optionnelle
            default_value = None
            if self.consume(TokenType.ASSIGN):
                default_value = self.parse_expression()
            
            parameters.append(Parameter(param_name, param_type, default_value))
            
            if not self.consume(TokenType.COMMA):
                break
        
        self.expect(TokenType.RPAREN)
        
        # Type de retour optionnel
        return_type = None
        if self.consume(TokenType.ARROW):
            return_type = self.parse_type()
        
        # Corps
        body = self.parse_block()
        
        return FunctionDecl(name, parameters, return_type, body, annotations or [])
    
    def parse_transaction_decl(self, annotations: List[Annotation] = None) -> 'TransactionDecl':
        """
        Parse: transaction name(params) { body } [rollback { ... }]
        
        Examples:
            transaction append_episode(record: EpisodicRecord) -> str {
                let id = episodic_append(record)
                audit_log("episode_added", id)
                return id
            }
            
            transaction update_concept(concept_id: str, new_vec: Vec) {
                let old = semantic_query(concept_id, k: 1)[0]
                semantic_upsert(concept_id, new_vec)
                audit_log("concept_updated", concept_id)
            } rollback {
                semantic_upsert(concept_id, old.centroid_vec)
            }
        """
        from parser.ast_nodes import TransactionDecl
        
        self.expect(TokenType.TRANSACTION)
        name = self.expect(TokenType.IDENTIFIER).value
        
        # Paramètres
        self.expect(TokenType.LPAREN)
        parameters = []
        
        while not self.match(TokenType.RPAREN):
            param_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            param_type = self.parse_type()
            
            # Valeur par défaut optionnelle
            default_value = None
            if self.consume(TokenType.ASSIGN):
                default_value = self.parse_expression()
            
            parameters.append(Parameter(param_name, param_type, default_value))
            
            if not self.consume(TokenType.COMMA):
                break
        
        self.expect(TokenType.RPAREN)
        
        # Type de retour optionnel
        return_type = None
        if self.consume(TokenType.ARROW):
            return_type = self.parse_type()
        
        # Corps principal
        body = self.parse_block()
        
        # Bloc rollback optionnel
        rollback_block = None
        if self.current_token().type == TokenType.IDENTIFIER and self.current_token().value == "rollback":
            self.advance()
            rollback_block = self.parse_block()
        
        # Modifiers (pour futures extensions: atomic, distributed, etc.)
        modifiers = []
        if annotations:
            for ann in annotations:
                if ann.name in ["atomic", "distributed", "isolated"]:
                    modifiers.append(ann.name)
        
        return TransactionDecl(name, parameters, body, rollback_block, modifiers)
    
    def parse_type_decl(self) -> TypeDecl:
        """Parse: type Name = Type"""
        self.expect(TokenType.TYPE)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        type_def = self.parse_type()
        
        return TypeDecl(name, type_def)
    
    def parse_return_stmt(self) -> ReturnStmt:
        """Parse: return expr"""
        self.expect(TokenType.RETURN)
        
        # return sans valeur
        if self.match(TokenType.RBRACE, TokenType.EOF):
            return ReturnStmt(None)
        
        value = self.parse_expression()
        return ReturnStmt(value)
    
    def parse_import_stmt(self) -> 'ImportStmt':
        """
        Parse: import module_name [as alias]
        
        Examples:
            import vectors
            import memory as mem
        """
        from parser.ast_nodes import ImportStmt
        
        self.expect(TokenType.IMPORT)
        module_name = self.expect(TokenType.IDENTIFIER).value
        
        alias = None
        if self.consume(TokenType.AS):
            alias = self.expect(TokenType.IDENTIFIER).value
        
        return ImportStmt(module_name, alias)
    
    def parse_if_stmt(self) -> IfStmt:
        """Parse: if condition { } else { }"""
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        then_block = self.parse_block()
        
        else_block = None
        if self.consume(TokenType.ELSE):
            if self.match(TokenType.IF):
                # else if -> transformer en block avec if
                else_if = self.parse_if_stmt()
                else_block = Block([else_if])
            else:
                else_block = self.parse_block()
        
        return IfStmt(condition, then_block, else_block)
    
    def parse_for_stmt(self) -> ForStmt:
        """Parse: for item in iterable { }"""
        self.expect(TokenType.FOR)
        variable = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.IN)
        iterable = self.parse_expression()
        body = self.parse_block()
        
        return ForStmt(variable, iterable, body)
    
    def parse_match_stmt(self) -> MatchStmt:
        """
        Parse: match value { case pattern -> { } case pattern where cond -> { } }
        """
        self.expect(TokenType.MATCH)
        value = self.parse_expression()
        self.expect(TokenType.LBRACE)
        
        cases = []
        while not self.match(TokenType.RBRACE):
            # Parse un case
            self.expect(TokenType.CASE)
            pattern = self.parse_pattern()
            
            # Condition where optionnelle
            condition = None
            if self.consume(TokenType.WHERE):
                condition = self.parse_expression()
            
            # Arrow et body
            self.expect(TokenType.ARROW)
            body = self.parse_block()
            
            cases.append(MatchCase(pattern, condition, body))
        
        self.expect(TokenType.RBRACE)
        return MatchStmt(value, cases)
    
    def parse_pattern(self) -> Pattern:
        """
        Parse un pattern:
        - Wildcard: _
        - Literal: 42, "hello", true, false
        - Type extraction: int(x), float(f), str(s)
        - Simple binding: x
        """
        token = self.current_token()
        
        # Wildcard
        if token.type == TokenType.IDENTIFIER and token.value == "_":
            self.advance()
            return WildcardPattern()
        
        # Literal bool
        if token.type in (TokenType.TRUE, TokenType.FALSE):
            self.advance()
            value = token.type == TokenType.TRUE
            return LiteralPattern(value)
        
        # Literal int
        if token.type == TokenType.INT_LITERAL:
            self.advance()
            return LiteralPattern(int(token.value))
        
        # Literal float
        if token.type == TokenType.FLOAT_LITERAL:
            self.advance()
            return LiteralPattern(float(token.value))
        
        # Literal string
        if token.type == TokenType.STRING_LITERAL:
            self.advance()
            return LiteralPattern(token.value)
        
        # Type keywords (int, float, str, bool) pour type extraction
        if token.type in (TokenType.INT_TYPE, TokenType.FLOAT_TYPE, TokenType.STR_TYPE, TokenType.BOOL_TYPE):
            type_name = token.value  # "int", "float", "str", "bool"
            self.advance()
            
            # Type extraction: int(x), float(f), etc.
            if self.consume(TokenType.LPAREN):
                inner_name = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.RPAREN)
                return IdentifierPattern(type_name, inner_name)
            
            # Juste le type sans extraction (invalide)
            raise Exception(f"Type {type_name} without variable binding at line {token.line}")
        
        # Identifier (type extraction custom ou simple binding)
        if token.type == TokenType.IDENTIFIER:
            name = token.value
            self.advance()
            
            # Type extraction: EpisodicRecord(e), Concept(c), etc.
            if self.consume(TokenType.LPAREN):
                inner_name = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.RPAREN)
                return IdentifierPattern(name, inner_name)
            
            # Simple binding: x
            return IdentifierPattern(name)
        
        raise Exception(f"Expected pattern, got {token.type.name} at line {token.line}, column {token.column}")
    
    def parse_while_stmt(self) -> WhileStmt:
        """Parse: while condition { }"""
        self.expect(TokenType.WHILE)
        condition = self.parse_expression()
        body = self.parse_block()
        
        return WhileStmt(condition, body)
    
    def parse_annotation(self) -> Annotation:
        """Parse: @name(arg1: val1, arg2: val2)"""
        token = self.expect(TokenType.ANNOTATION)
        name = token.value[1:]  # Enlever le @
        
        arguments = {}
        if self.consume(TokenType.LPAREN):
            while not self.match(TokenType.RPAREN):
                arg_name = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.COLON)
                arg_value = self.parse_expression()
                arguments[arg_name] = arg_value
                
                if not self.consume(TokenType.COMMA):
                    break
            
            self.expect(TokenType.RPAREN)
        
        return Annotation(name, arguments)
    
    def parse_statement(self) -> Statement:
        """Parse un statement"""
        # Annotations (pour fonctions)
        annotations = []
        while self.match(TokenType.ANNOTATION):
            annotations.append(self.parse_annotation())
        
        token = self.current_token()
        
        # Import (Phase 3.2)
        if token.type == TokenType.IMPORT:
            return self.parse_import_stmt()
        
        # Déclarations
        if token.type in (TokenType.LET, TokenType.CONST):
            return self.parse_var_decl()
        
        if token.type == TokenType.FN:
            return self.parse_function_decl(annotations)
        
        if token.type == TokenType.TRANSACTION:
            return self.parse_transaction_decl(annotations)
        
        if token.type == TokenType.TYPE:
            return self.parse_type_decl()
        
        # Control flow
        if token.type == TokenType.RETURN:
            return self.parse_return_stmt()
        
        if token.type == TokenType.IF:
            return self.parse_if_stmt()
        
        if token.type == TokenType.FOR:
            return self.parse_for_stmt()
        
        if token.type == TokenType.WHILE:
            return self.parse_while_stmt()
        
        if token.type == TokenType.MATCH:
            return self.parse_match_stmt()
        
        # Expression statement
        expr = self.parse_expression()
        return ExpressionStmt(expr)
    
    # ============================================
    # Program
    # ============================================
    
    def parse(self) -> Program:
        """Parse le programme complet"""
        statements = []
        
        while not self.match(TokenType.EOF):
            statements.append(self.parse_statement())
            # Consommer les points-virgules optionnels après chaque statement
            self.consume(TokenType.SEMICOLON)
        
        return Program(statements)


# ============================================
# Tests
# ============================================

if __name__ == '__main__':
    from .lexer import Lexer
    
    print("=== Test Parser ===\n")
    
    # Test 1: Variable
    code1 = "let x: int = 42"
    print(f"Code: {code1}")
    lexer1 = Lexer(code1)
    parser1 = Parser(lexer1.tokenize())
    ast1 = parser1.parse()
    print(f"AST: {ast1}")
    print(f"Statement: {ast1.statements[0]}\n")
    
    # Test 2: Fonction
    code2 = """
    fn add(a: int, b: int) -> int {
        return a + b
    }
    """
    print(f"Code: {code2}")
    lexer2 = Lexer(code2)
    parser2 = Parser(lexer2.tokenize())
    ast2 = parser2.parse()
    print(f"AST: {ast2}")
    print(f"Statement: {ast2.statements[0]}\n")
    
    # Test 3: Type déclaration
    code3 = "type Vec = Vector<float, dim=256, q=8>"
    print(f"Code: {code3}")
    lexer3 = Lexer(code3)
    parser3 = Parser(lexer3.tokenize())
    ast3 = parser3.parse()
    print(f"AST: {ast3}")
    print(f"Statement: {ast3.statements[0]}\n")
    
    # Test 4: Annotation + fonction
    code4 = """
    @plastic(rate: 0.001)
    fn adapt(state: Vec, delta: Vec) -> Vec {
        return state
    }
    """
    print(f"Code: {code4}")
    lexer4 = Lexer(code4)
    parser4 = Parser(lexer4.tokenize())
    ast4 = parser4.parse()
    print(f"AST: {ast4}")
    print(f"Statement: {ast4.statements[0]}\n")
    
    print("✅ Parser tests passed!")
