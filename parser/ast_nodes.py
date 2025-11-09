"""
NORMiL AST Nodes (Abstract Syntax Tree)
========================================

Définition des nœuds de l'arbre syntaxique abstrait pour NORMiL.

Hiérarchie:
- ASTNode (base)
  - Expression
    - Literal (int, float, str, bool)
    - Identifier
    - BinaryOp
    - UnaryOp
    - FunctionCall
    - FieldAccess
    - IndexAccess
    - ListLiteral
    - MapLiteral
    - StructLiteral
  - Statement
    - VarDecl
    - FunctionDecl
    - TypeDecl
    - ReturnStmt
    - IfStmt
    - ForStmt
    - WhileStmt
    - MatchStmt
    - TransactionDecl
    - ExpressionStmt
  - Type
    - PrimitiveType
    - VectorType
    - StructType
    - ListType
    - MapType
    - OptionalType
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from abc import ABC, abstractmethod


# ============================================
# Base Classes
# ============================================

class ASTNode(ABC):
    """Classe de base pour tous les nœuds AST"""
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


# ============================================
# Types
# ============================================

@dataclass
class Type(ASTNode):
    """Classe de base pour les types"""
    pass


@dataclass
class PrimitiveType(Type):
    """Type primitif: int, float, bool, str, timestamp, uuid"""
    name: str  # "int", "float", "bool", "str", "timestamp", "uuid"
    
    def __repr__(self) -> str:
        return f"PrimitiveType({self.name})"


@dataclass
class VectorType(Type):
    """Type vectoriel: Vector<float, dim=256, q=8>"""
    element_type: Type
    dim: int
    quantization: Optional[int] = None  # Bits de quantisation (optionnel)
    
    def __repr__(self) -> str:
        q_str = f", q={self.quantization}" if self.quantization else ""
        return f"Vector<{self.element_type}, dim={self.dim}{q_str}>"


@dataclass
class StructType(Type):
    """Type structure: {field1: Type1, field2: Type2}"""
    fields: Dict[str, Type]
    
    def __repr__(self) -> str:
        fields_str = ", ".join(f"{k}: {v}" for k, v in self.fields.items())
        return f"{{{fields_str}}}"


@dataclass
class ListType(Type):
    """Type liste: list<Type>"""
    element_type: Type
    
    def __repr__(self) -> str:
        return f"list<{self.element_type}>"


@dataclass
class MapType(Type):
    """Type map: map<KeyType, ValueType>"""
    key_type: Type
    value_type: Type
    
    def __repr__(self) -> str:
        return f"map<{self.key_type}, {self.value_type}>"


@dataclass
class OptionalType(Type):
    """Type optionnel: optional<Type>"""
    inner_type: Type
    
    def __repr__(self) -> str:
        return f"optional<{self.inner_type}>"


@dataclass
class NamedType(Type):
    """Type nommé (défini par utilisateur ou alias)"""
    name: str
    
    def __repr__(self) -> str:
        return f"NamedType({self.name})"


# ============================================
# Expressions
# ============================================

@dataclass
class Expression(ASTNode):
    """Classe de base pour les expressions"""
    pass


@dataclass
class IntLiteral(Expression):
    """Littéral entier: 42"""
    value: int
    
    def __repr__(self) -> str:
        return f"IntLiteral({self.value})"


@dataclass
class FloatLiteral(Expression):
    """Littéral flottant: 3.14"""
    value: float
    
    def __repr__(self) -> str:
        return f"FloatLiteral({self.value})"


@dataclass
class StringLiteral(Expression):
    """Littéral chaîne: "hello" """
    value: str
    
    def __repr__(self) -> str:
        return f"StringLiteral({self.value!r})"


@dataclass
class BoolLiteral(Expression):
    """Littéral booléen: true / false"""
    value: bool
    
    def __repr__(self) -> str:
        return f"BoolLiteral({self.value})"


@dataclass
class Identifier(Expression):
    """Identificateur: variable, fonction, type"""
    name: str
    
    def __repr__(self) -> str:
        return f"Identifier({self.name})"


@dataclass
class BinaryOp(Expression):
    """Opération binaire: left op right"""
    operator: str  # "+", "-", "*", "/", "==", "&&", etc.
    left: Expression
    right: Expression
    
    def __repr__(self) -> str:
        return f"BinaryOp({self.operator}, {self.left}, {self.right})"


@dataclass
class UnaryOp(Expression):
    """Opération unaire: op expr"""
    operator: str  # "-", "!"
    operand: Expression
    
    def __repr__(self) -> str:
        return f"UnaryOp({self.operator}, {self.operand})"


@dataclass
class NamedArgument(Expression):
    """Argument nommé: name: value"""
    name: str
    value: Expression
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.value}"


@dataclass
class FunctionCall(Expression):
    """Appel de fonction: func(arg1, arg2, name: value)"""
    function: Expression  # Peut être Identifier ou FieldAccess
    arguments: List[Expression]  # Peut inclure NamedArgument
    
    def __repr__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"FunctionCall({self.function}, [{args_str}])"


@dataclass
class FieldAccess(Expression):
    """Accès champ: object.field"""
    object: Expression
    field: str
    
    def __repr__(self) -> str:
        return f"FieldAccess({self.object}.{self.field})"


@dataclass
class IndexAccess(Expression):
    """Accès index: array[index]"""
    object: Expression
    index: Expression
    
    def __repr__(self) -> str:
        return f"IndexAccess({self.object}[{self.index}])"


@dataclass
class ListLiteral(Expression):
    """Littéral liste: [elem1, elem2]"""
    elements: List[Expression]
    
    def __repr__(self) -> str:
        elems_str = ", ".join(str(e) for e in self.elements)
        return f"ListLiteral([{elems_str}])"


@dataclass
class MapLiteral(Expression):
    """Littéral map: {key1: val1, key2: val2}"""
    entries: List[tuple[Expression, Expression]]
    
    def __repr__(self) -> str:
        entries_str = ", ".join(f"{k}: {v}" for k, v in self.entries)
        return f"MapLiteral({{{entries_str}}})"


@dataclass
class StructLiteral(Expression):
    """Littéral structure: TypeName {field1: val1, field2: val2}"""
    type_name: Optional[str]  # Peut être None pour struct anonyme
    fields: Dict[str, Expression]
    
    def __repr__(self) -> str:
        fields_str = ", ".join(f"{k}: {v}" for k, v in self.fields.items())
        type_prefix = f"{self.type_name} " if self.type_name else ""
        return f"StructLiteral({type_prefix}{{{fields_str}}})"


# ============================================
# Statements
# ============================================

@dataclass
class Statement(ASTNode):
    """Classe de base pour les statements"""
    pass


@dataclass
class VarDecl(Statement):
    """Déclaration de variable: let x: int = 42"""
    name: str
    var_type: Optional[Type]  # None si inféré
    value: Expression
    is_const: bool = False  # True pour const
    
    def __repr__(self) -> str:
        keyword = "const" if self.is_const else "let"
        type_str = f": {self.var_type}" if self.var_type else ""
        return f"VarDecl({keyword} {self.name}{type_str} = {self.value})"


@dataclass
class Parameter:
    """Paramètre de fonction"""
    name: str
    param_type: Type
    default_value: Optional[Expression] = None
    
    def __repr__(self) -> str:
        default_str = f" = {self.default_value}" if self.default_value else ""
        return f"{self.name}: {self.param_type}{default_str}"


@dataclass
class Annotation:
    """Annotation: @name(arg1: val1, arg2: val2)"""
    name: str
    arguments: Dict[str, Expression]
    
    def __repr__(self) -> str:
        args_str = ", ".join(f"{k}: {v}" for k, v in self.arguments.items())
        return f"@{self.name}({args_str})"


@dataclass
class FunctionDecl(Statement):
    """Déclaration de fonction"""
    name: str
    parameters: List[Parameter]
    return_type: Optional[Type]
    body: 'Block'
    annotations: List[Annotation] = field(default_factory=list)
    
    def __repr__(self) -> str:
        annot_str = " ".join(str(a) for a in self.annotations)
        params_str = ", ".join(str(p) for p in self.parameters)
        ret_str = f" -> {self.return_type}" if self.return_type else ""
        return f"{annot_str} fn {self.name}({params_str}){ret_str} {{ ... }}"


@dataclass
class Block(ASTNode):
    """Bloc de statements: { stmt1; stmt2; }"""
    statements: List[Statement]
    
    def __repr__(self) -> str:
        return f"Block({len(self.statements)} statements)"


@dataclass
class TypeDecl(Statement):
    """Déclaration de type: type Vec = Vector<float, dim=256>"""
    name: str
    type_def: Type
    
    def __repr__(self) -> str:
        return f"TypeDecl(type {self.name} = {self.type_def})"


@dataclass
class ImportStmt(Statement):
    """Statement import: import module_name [as alias]"""
    module_name: str
    alias: Optional[str] = None
    
    def __repr__(self) -> str:
        if self.alias:
            return f"ImportStmt(import {self.module_name} as {self.alias})"
        return f"ImportStmt(import {self.module_name})"


@dataclass
class ReturnStmt(Statement):
    """Statement return: return expr"""
    value: Optional[Expression]
    
    def __repr__(self) -> str:
        val_str = str(self.value) if self.value else ""
        return f"ReturnStmt({val_str})"


@dataclass
class IfStmt(Statement):
    """Statement if: if condition { ... } else { ... }"""
    condition: Expression
    then_block: Block
    else_block: Optional[Block]
    
    def __repr__(self) -> str:
        else_str = " else { ... }" if self.else_block else ""
        return f"IfStmt(if {self.condition} {{ ... }}{else_str})"


@dataclass
class ForStmt(Statement):
    """Statement for: for item in iterable { ... }"""
    variable: str
    iterable: Expression
    body: Block
    
    def __repr__(self) -> str:
        return f"ForStmt(for {self.variable} in {self.iterable} {{ ... }})"


@dataclass
class WhileStmt(Statement):
    """Statement while: while condition { ... }"""
    condition: Expression
    body: Block
    
    def __repr__(self) -> str:
        return f"WhileStmt(while {self.condition} {{ ... }})"


@dataclass
class Pattern(ASTNode):
    """Classe de base pour les patterns"""
    pass


@dataclass
class IdentifierPattern(Pattern):
    """Pattern identificateur: case int(x) ou case x"""
    name: str
    inner_name: Optional[str] = None  # Pour case int(x), name="int", inner="x"
    
    def __repr__(self) -> str:
        if self.inner_name:
            return f"{self.name}({self.inner_name})"
        return self.name


@dataclass
class LiteralPattern(Pattern):
    """Pattern littéral: case 42 ou case "hello" """
    value: Any
    
    def __repr__(self) -> str:
        return f"LiteralPattern({self.value})"


@dataclass
class WildcardPattern(Pattern):
    """Pattern wildcard: case _"""
    
    def __repr__(self) -> str:
        return "_"


@dataclass
class MatchCase:
    """Case dans un match"""
    pattern: Pattern
    condition: Optional[Expression]  # where clause
    body: Block
    
    def __repr__(self) -> str:
        where_str = f" where {self.condition}" if self.condition else ""
        return f"case {self.pattern}{where_str} -> {{ ... }}"


@dataclass
class MatchStmt(Statement):
    """Statement match: match value { case pattern -> ... }"""
    value: Expression
    cases: List[MatchCase]
    
    def __repr__(self) -> str:
        return f"MatchStmt(match {self.value} {{ {len(self.cases)} cases }})"


@dataclass
class TransactionDecl(Statement):
    """Déclaration de transaction"""
    name: str
    parameters: List[Parameter]
    body: Block
    rollback_block: Optional[Block] = None
    modifiers: List[str] = field(default_factory=list)  # ["atomic"], ["distributed"], etc.
    
    def __repr__(self) -> str:
        mods = " ".join(self.modifiers)
        params_str = ", ".join(str(p) for p in self.parameters)
        return f"{mods} transaction {self.name}({params_str}) {{ ... }}"


@dataclass
class ExpressionStmt(Statement):
    """Statement expression: expr;"""
    expression: Expression
    
    def __repr__(self) -> str:
        return f"ExpressionStmt({self.expression})"


@dataclass
class PrimitiveDecl(Statement):
    """Déclaration de primitive: primitive name(params) -> Type"""
    name: str
    parameters: List[Parameter]
    return_type: Type
    
    def __repr__(self) -> str:
        params_str = ", ".join(str(p) for p in self.parameters)
        return f"primitive {self.name}({params_str}) -> {self.return_type}"


# ============================================
# Program
# ============================================

@dataclass
class Program(ASTNode):
    """Programme complet (module)"""
    statements: List[Statement]
    
    def __repr__(self) -> str:
        return f"Program({len(self.statements)} top-level statements)"


# ============================================
# Helpers pour construction AST
# ============================================

def make_int(value: int) -> IntLiteral:
    """Helper pour créer un IntLiteral"""
    return IntLiteral(value)


def make_float(value: float) -> FloatLiteral:
    """Helper pour créer un FloatLiteral"""
    return FloatLiteral(value)


def make_string(value: str) -> StringLiteral:
    """Helper pour créer un StringLiteral"""
    return StringLiteral(value)


def make_bool(value: bool) -> BoolLiteral:
    """Helper pour créer un BoolLiteral"""
    return BoolLiteral(value)


def make_ident(name: str) -> Identifier:
    """Helper pour créer un Identifier"""
    return Identifier(name)


def make_binop(op: str, left: Expression, right: Expression) -> BinaryOp:
    """Helper pour créer un BinaryOp"""
    return BinaryOp(op, left, right)


if __name__ == '__main__':
    # Test AST construction
    print("=== Test AST Nodes ===\n")
    
    # let x: int = 42
    var_decl = VarDecl(
        name="x",
        var_type=PrimitiveType("int"),
        value=IntLiteral(42),
        is_const=False
    )
    print(f"1. {var_decl}\n")
    
    # fn add(a: int, b: int) -> int { return a + b }
    func_decl = FunctionDecl(
        name="add",
        parameters=[
            Parameter("a", PrimitiveType("int")),
            Parameter("b", PrimitiveType("int"))
        ],
        return_type=PrimitiveType("int"),
        body=Block([
            ReturnStmt(
                BinaryOp("+", Identifier("a"), Identifier("b"))
            )
        ])
    )
    print(f"2. {func_decl}\n")
    
    # type Vec = Vector<float, dim=256, q=8>
    type_decl = TypeDecl(
        name="Vec",
        type_def=VectorType(
            element_type=PrimitiveType("float"),
            dim=256,
            quantization=8
        )
    )
    print(f"3. {type_decl}\n")
    
    # @plastic(rate: 0.001)
    annotation = Annotation(
        name="plastic",
        arguments={"rate": FloatLiteral(0.001)}
    )
    print(f"4. {annotation}\n")
    
    print("✅ AST nodes test passed!")
