"""
NORMiL Executor
===============

Exécuteur d'AST NORMiL.

L'executor parcourt l'arbre syntaxique abstrait (AST) et exécute le code.

Fonctionnalités:
- Évaluation d'expressions
- Exécution de statements
- Gestion des variables et fonctions
- Appel de primitives natives
- Gestion de la portée (scope)
- Système d'imports de modules (Phase 3.2)
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from parser.ast_nodes import *
from runtime.primitives import PRIMITIVES
from runtime.normil_types import Vec, EpisodicRecord, Concept, ProtoInstinct, SparseVec


class ExecutionError(Exception):
    """Erreur d'exécution"""
    pass


class ReturnValue(Exception):
    """Exception pour gérer les return dans les fonctions"""
    def __init__(self, value: Any):
        self.value = value


class Scope:
    """Portée des variables"""
    def __init__(self, parent: Optional['Scope'] = None):
        self.parent = parent
        self.variables: Dict[str, Any] = {}
        self.functions: Dict[str, FunctionDecl] = {}
        self.transactions: Dict[str, 'TransactionDecl'] = {}  # Phase 6.4
        self.types: Dict[str, Type] = {}
        self.modules: Dict[str, 'Scope'] = {}  # Modules importés
        self._python_module: Optional[Any] = None  # Pour Phase 4.1: référence au module Python si c'est un wrapper
    
    def define_var(self, name: str, value: Any):
        """Définit une variable dans la portée actuelle"""
        self.variables[name] = value
    
    def get_var(self, name: str) -> Any:
        """Récupère une variable (cherche dans les portées parentes)"""
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get_var(name)
        raise ExecutionError(f"Undefined variable: {name}")
    
    def set_var(self, name: str, value: Any):
        """Modifie une variable existante"""
        if name in self.variables:
            self.variables[name] = value
        elif self.parent:
            self.parent.set_var(name, value)
        else:
            raise ExecutionError(f"Undefined variable: {name}")
    
    def define_function(self, name: str, func: FunctionDecl):
        """Définit une fonction"""
        self.functions[name] = func
    
    def get_function(self, name: str) -> Optional[FunctionDecl]:
        """Récupère une fonction"""
        if name in self.functions:
            return self.functions[name]
        if self.parent:
            return self.parent.get_function(name)
        return None
    
    def define_transaction(self, name: str, transaction: 'TransactionDecl'):
        """Définit une transaction"""
        self.transactions[name] = transaction
    
    def get_transaction(self, name: str) -> Optional['TransactionDecl']:
        """Récupère une transaction"""
        if name in self.transactions:
            return self.transactions[name]
        if self.parent:
            return self.parent.get_transaction(name)
        return None
    
    def define_type(self, name: str, type_def: Type):
        """Définit un type"""
        self.types[name] = type_def
    
    def get_type(self, name: str) -> Optional[Type]:
        """Récupère un type"""
        if name in self.types:
            return self.types[name]
        if self.parent:
            return self.parent.get_type(name)
        return None
    
    def define_module(self, name: str, module_scope: 'Scope'):
        """Définit un module importé"""
        self.modules[name] = module_scope
    
    def get_module(self, name: str) -> Optional['Scope']:
        """Récupère un module"""
        if name in self.modules:
            return self.modules[name]
        if self.parent:
            return self.parent.get_module(name)
        return None


class Executor:
    """Exécuteur de code NORMiL"""
    
    def __init__(self):
        self.global_scope = Scope()
        self.current_scope = self.global_scope
        
        # Métadonnées pour les annotations
        self.function_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Chemin de recherche des modules (Phase 3.2)
        self.module_paths: List[Path] = [
            Path.cwd() / "modules",  # modules/ dans le dossier courant
            Path.cwd(),  # dossier courant
        ]
        
        # Cache des modules chargés (éviter re-chargement)
        self.loaded_modules: Dict[str, Scope] = {}
        
        # Enregistrer les primitives dans le scope global
        for name, func in PRIMITIVES.items():
            self.global_scope.variables[name] = func
    
    # ============================================
    # Expressions
    # ============================================
    
    def eval_expression(self, expr: Expression) -> Any:
        """Évalue une expression"""
        
        # Littéraux
        if isinstance(expr, IntLiteral):
            return expr.value
        
        if isinstance(expr, FloatLiteral):
            return expr.value
        
        if isinstance(expr, StringLiteral):
            return expr.value
        
        if isinstance(expr, BoolLiteral):
            return expr.value
        
        # Identificateur (variable)
        if isinstance(expr, Identifier):
            return self.current_scope.get_var(expr.name)
        
        # Opérations binaires
        if isinstance(expr, BinaryOp):
            return self.eval_binary_op(expr)
        
        # Opérations unaires
        if isinstance(expr, UnaryOp):
            return self.eval_unary_op(expr)
        
        # Appel de fonction
        if isinstance(expr, FunctionCall):
            return self.eval_function_call(expr)
        
        # Accès champ
        if isinstance(expr, FieldAccess):
            # Vérifier si c'est un accès à un module (Phase 3.2 + Phase 4.1)
            if isinstance(expr.object, Identifier):
                module = self.current_scope.get_module(expr.object.name)
                if module:
                    # PHASE 4.1: Support des modules Python
                    if hasattr(module, '_python_module') and module._python_module is not None:
                        py_module = module._python_module
                        if hasattr(py_module, expr.field):
                            return getattr(py_module, expr.field)
                        raise ExecutionError(
                            f"Python module '{expr.object.name}' has no attribute '{expr.field}'"
                        )
                    
                    # Modules NORMiL
                    # Accès à une fonction du module
                    func = module.get_function(expr.field)
                    if func:
                        return func  # Retourner la FunctionDecl
                    # Accès à une variable du module
                    try:
                        return module.get_var(expr.field)
                    except:
                        raise ExecutionError(
                            f"Module '{expr.object.name}' has no attribute '{expr.field}'"
                        )
            
            # Accès standard à un champ d'objet
            obj = self.eval_expression(expr.object)
            
            # Support dict/map comme struct (Phase 3.5)
            if isinstance(obj, dict):
                if expr.field in obj:
                    return obj[expr.field]
                raise ExecutionError(f"Struct has no field: {expr.field}")
            
            # Support attributs Python natifs
            if hasattr(obj, expr.field):
                return getattr(obj, expr.field)
            
            raise ExecutionError(f"Object has no field: {expr.field}")
        
        # Accès index
        if isinstance(expr, IndexAccess):
            obj = self.eval_expression(expr.object)
            index = self.eval_expression(expr.index)
            return obj[index]
        
        # Liste littérale
        if isinstance(expr, ListLiteral):
            return [self.eval_expression(elem) for elem in expr.elements]
        
        # Map littérale
        if isinstance(expr, MapLiteral):
            result = {}
            for key_expr, val_expr in expr.entries:
                key = self.eval_expression(key_expr)
                val = self.eval_expression(val_expr)
                result[key] = val
            return result
        
        # Struct littérale
        if isinstance(expr, StructLiteral):
            # Évaluer les champs
            fields = {}
            for field_name, field_expr in expr.fields.items():
                fields[field_name] = self.eval_expression(field_expr)
            
            # Si le struct a un nom de type, créer l'objet approprié
            if expr.type_name:
                if expr.type_name == "EpisodicRecord":
                    return EpisodicRecord(**fields)
                elif expr.type_name == "Concept":
                    return Concept(**fields)
                elif expr.type_name == "ProtoInstinct":
                    return ProtoInstinct(**fields)
                elif expr.type_name == "SparseVec":
                    return SparseVec(**fields)
                # Autres types à ajouter ici
                else:
                    # Type inconnu, retourner dict
                    return fields
            
            # Struct anonyme: retourner un dict
            return fields
        
        raise ExecutionError(f"Unknown expression type: {type(expr)}")
    
    def eval_binary_op(self, expr: BinaryOp) -> Any:
        """Évalue une opération binaire"""
        op = expr.operator
        
        # Affectation - traiter AVANT d'évaluer left (car left est un lvalue)
        if op == '=':
            # Évaluer seulement right
            right = self.eval_expression(expr.right)
            
            if isinstance(expr.left, Identifier):
                # Essayer de modifier une variable existante d'abord (cherche dans scopes parents)
                # Sinon, créer une nouvelle variable dans le scope actuel
                var_name = expr.left.name
                try:
                    self.current_scope.set_var(var_name, right)
                except ExecutionError:
                    # Variable n'existe pas, la créer
                    self.current_scope.define_var(var_name, right)
                return right
            elif isinstance(expr.left, FieldAccess):
                # Assignation à un champ: obj.field = value
                obj = self.eval_expression(expr.left.object)
                field_name = expr.left.field
                
                # Support dict (structs anonymes)
                if isinstance(obj, dict):
                    obj[field_name] = right
                    return right
                
                # Support objets Python avec setattr
                if hasattr(obj, field_name):
                    setattr(obj, field_name, right)
                    return right
                
                raise ExecutionError(f"Cannot assign to field '{field_name}' of {type(obj).__name__}")
            elif isinstance(expr.left, IndexAccess):
                # Assignation à un index: arr[i] = value
                obj = self.eval_expression(expr.left.object)
                index = self.eval_expression(expr.left.index)
                if isinstance(obj, (list, dict)):
                    obj[index] = right
                    return right
                else:
                    raise ExecutionError(f"Cannot assign to index of non-indexable object")
            raise ExecutionError("Can only assign to variables, fields, or indices")
        
        if op == '+=':
            # Évaluer right, mais left doit exister
            if isinstance(expr.left, Identifier):
                current = self.current_scope.get_var(expr.left.name)
                right = self.eval_expression(expr.right)
                new_value = current + right
                self.current_scope.set_var(expr.left.name, new_value)
                return new_value
            raise ExecutionError("Can only assign to variables")
        
        # Pour tous les autres opérateurs, évaluer left et right normalement
        left = self.eval_expression(expr.left)
        right = self.eval_expression(expr.right)
        
        # Arithmétiques
        if op == '+':
            # String concatenation (Phase 3.3)
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            return left + right
        if op == '-':
            return left - right
        if op == '*':
            return left * right
        if op == '/':
            return left / right
        if op == '%':
            return left % right
        
        # Vectoriels (déléguer aux primitives)
        if op == '.+':
            from runtime.primitives import vec_add
            return vec_add(left, right)
        if op == '.-':
            from runtime.primitives import vec_sub
            return vec_sub(left, right)
        if op == '.*':
            from runtime.primitives import vec_mul
            return vec_mul(left, right)
        if op == '@':
            from runtime.primitives import dot
            return dot(left, right)
        
        # Comparaisons
        if op == '==':
            return left == right
        if op == '!=':
            return left != right
        if op == '<':
            return left < right
        if op == '>':
            return left > right
        if op == '<=':
            return left <= right
        if op == '>=':
            return left >= right
        
        # Logiques
        if op == '&&':
            return left and right
        if op == '||':
            return left or right
        
        # Autres opérateurs d'affectation similaires
        
        raise ExecutionError(f"Unknown binary operator: {op}")
    
    def eval_unary_op(self, expr: UnaryOp) -> Any:
        """Évalue une opération unaire"""
        operand = self.eval_expression(expr.operand)
        op = expr.operator
        
        if op == '-':
            return -operand
        if op == '!':
            return not operand
        
        raise ExecutionError(f"Unknown unary operator: {op}")
    
    def eval_function_call(self, expr: FunctionCall) -> Any:
        """Évalue un appel de fonction"""
        from parser.ast_nodes import NamedArgument
        
        # Séparer arguments positionnels et nommés
        positional_args = []
        named_args = {}
        
        for arg in expr.arguments:
            if isinstance(arg, NamedArgument):
                # Argument nommé
                named_args[arg.name] = self.eval_expression(arg.value)
            else:
                # Argument positionnel
                positional_args.append(self.eval_expression(arg))
        
        # Récupérer la fonction
        if isinstance(expr.function, Identifier):
            func_name = expr.function.name
            
            # Vérifier si c'est une primitive
            if func_name in PRIMITIVES:
                # Appeler avec args et kwargs
                return PRIMITIVES[func_name](*positional_args, **named_args)
            
            # Vérifier si c'est une fonction définie
            func_decl = self.current_scope.get_function(func_name)
            if func_decl:
                return self.call_user_function(func_decl, positional_args, named_args)
            
            # Vérifier si c'est une transaction (Phase 6.4)
            trans_decl = self.current_scope.get_transaction(func_name)
            if trans_decl:
                return self.call_transaction(trans_decl, positional_args, named_args)
            
            raise ExecutionError(f"Undefined function: {func_name}")
        
        # Si c'est déjà une FunctionDecl (ex: module.fonction)
        if isinstance(expr.function, FunctionDecl):
            return self.call_user_function(expr.function, positional_args, named_args)
        
        # Évaluer pour FieldAccess ou autres (ex: module.fonction())
        func = self.eval_expression(expr.function)
        
        # Si on a récupéré une FunctionDecl
        if isinstance(func, FunctionDecl):
            return self.call_user_function(func, positional_args, named_args)
        
        # Fonction callable Python
        if callable(func):
            return func(*positional_args, **named_args)
        
        raise ExecutionError(f"Not a function: {expr.function}")
    
    def call_user_function(self, func: FunctionDecl, positional_args: List[Any], named_args: dict = None) -> Any:
        """Appelle une fonction définie par l'utilisateur"""
        if named_args is None:
            named_args = {}
        
        # Traiter les annotations avant l'exécution
        metadata = self.process_annotations(func)
        
        # Vérifier si la fonction est atomique
        is_atomic = 'atomic' in metadata
        snapshot = None
        
        if is_atomic:
            # Créer un snapshot de l'état du scope actuel
            snapshot = self.create_scope_snapshot(self.current_scope)
        
        # Combiner arguments positionnels et nommés
        # Créer un mapping param_name -> value
        param_values = {}
        
        # D'abord les positionnels
        for i, arg in enumerate(positional_args):
            if i < len(func.parameters):
                param_values[func.parameters[i].name] = arg
        
        # Ensuite les nommés (peuvent override les positionnels)
        for name, value in named_args.items():
            # Vérifier que le paramètre existe
            param_names = [p.name for p in func.parameters]
            if name not in param_names:
                raise ExecutionError(
                    f"Function {func.name} has no parameter named '{name}'"
                )
            param_values[name] = value
        
        # Vérifier que tous les paramètres sont fournis
        if len(param_values) != len(func.parameters):
            raise ExecutionError(
                f"Function {func.name} expects {len(func.parameters)} arguments, "
                f"got {len(param_values)}"
            )
        
        # Créer un nouveau scope pour la fonction
        func_scope = Scope(parent=self.current_scope)
        
        # Injecter les métadonnées dans le scope (accessibles via des variables spéciales)
        if metadata:
            func_scope.define_var("__metadata__", metadata)
        
        # Lier les paramètres
        for param in func.parameters:
            func_scope.define_var(param.name, param_values[param.name])
        
        # Exécuter le corps de la fonction
        prev_scope = self.current_scope
        self.current_scope = func_scope
        
        # Phase 7.4: Gestion automatique de la plasticité
        plastic_config = metadata.get('plastic', None)
        weights_before = None
        
        try:
            # Si fonction plastique, capturer l'état initial
            if plastic_config and plastic_config.get('enabled'):
                # Chercher des variables qui pourraient être des poids
                # (par convention, variables nommées "weights", "w", "synapses", etc.)
                weight_candidates = ['weights', 'w', 'synapses', 'connections']
                for var_name in weight_candidates:
                    try:
                        val = func_scope.get_var(var_name)
                        if isinstance(val, Vec):
                            weights_before = val
                            break
                    except:
                        pass
            
            self.exec_block(func.body)
            # Pas de return explicite -> retourne None
            result = None
            
        except ReturnValue as ret:
            result = ret.value
        
        except Exception as e:
            # En cas d'erreur dans une fonction atomique, rollback
            if is_atomic and snapshot is not None:
                self.restore_scope_snapshot(self.current_scope, snapshot)
                raise ExecutionError(
                    f"Transaction rolled back in {func.name}: {str(e)}"
                ) from e
            else:
                raise
        
        finally:
            self.current_scope = prev_scope
        
        # Phase 7.4: Post-traitement plastique
        if plastic_config and plastic_config.get('enabled') and not plastic_config.get('is_stable'):
            from runtime.primitives import normalize_plasticity, compute_stability, decay_learning_rate, _plasticity_modes
            
            # Incrémenter le compteur d'étapes
            plastic_config['step_count'] = plastic_config.get('step_count', 0) + 1
            
            # Flag pour savoir si on doit appliquer le decay
            should_decay = False
            
            # Si on a capturé des poids ET un résultat vecteur, vérifier la stabilité
            if weights_before is not None and isinstance(result, Vec):
                is_stable = compute_stability(
                    weights_before, 
                    result, 
                    plastic_config.get('stability_threshold', 0.01)
                )
                
                if is_stable:
                    plastic_config['is_stable'] = True
                    # print(f"Function {func.name} reached stability after {plastic_config['step_count']} steps")
                else:
                    # Seulement décroître si on a vérifié la stabilité ET qu'on n'est pas stable
                    should_decay = True
            
            # Normaliser le résultat si le mode le requiert (via registry)
            mode = plastic_config.get('mode', '')
            if isinstance(result, Vec) and _plasticity_modes.should_normalize(mode):
                result = normalize_plasticity(result)
            
            # Decay automatique du learning rate seulement si instable
            # Utilise le decay_factor configuré (Phase 7.7)
            if should_decay and 'rate' in plastic_config:
                old_rate = plastic_config['rate']
                decay_factor = plastic_config.get('decay_factor', 0.99)
                plastic_config['rate'] = decay_learning_rate(old_rate, decay_factor)
        
        return result

    def process_annotations(self, func: FunctionDecl) -> Dict[str, Any]:
        """
        Traite les annotations d'une fonction et retourne les métadonnées.
        
        Annotations supportées:
        - @plastic(rate: float, mode: str, stability_threshold: float): Fonction avec plasticité neuronale
        - @atomic: Transaction avec rollback automatique
        - @memoize: Cache les résultats (future)
        - @profile: Profile l'exécution (future)
        
        Args:
            func: Déclaration de fonction avec annotations
        
        Returns:
            Dictionnaire de métadonnées
        """
        metadata = {}
        
        for annotation in func.annotations:
            if annotation.name == "plastic":
                # Annotation @plastic avec paramètres
                plastic_config = {}
                
                # Extraire les arguments
                for arg_name, arg_expr in annotation.arguments.items():
                    value = self.eval_expression(arg_expr)
                    plastic_config[arg_name] = value
                
                # Valeurs par défaut (Phase 7.7: ajout decay_factor)
                metadata['plastic'] = {
                    'rate': plastic_config.get('rate', 0.001),
                    'mode': plastic_config.get('mode', 'hebbian'),
                    'stability_threshold': plastic_config.get('stability_threshold', 0.01),
                    'decay_factor': plastic_config.get('decay_factor', 0.99),  # Phase 7.7
                    'enabled': True,
                    'step_count': 0,  # Compteur d'étapes pour tracking
                    'is_stable': False  # Indicateur de stabilité
                }
            
            elif annotation.name == "atomic":
                # Annotation @atomic : transaction avec rollback
                metadata['atomic'] = {
                    'enabled': True,
                    'isolation_level': 'serializable'
                }
            
            elif annotation.name == "memoize":
                # Future: cache de résultats
                metadata['memoize'] = {
                    'enabled': True,
                    'cache': {}
                }
            
            elif annotation.name == "profile":
                # Future: profiling
                metadata['profile'] = {
                    'enabled': True
                }
        
        # Stocker les métadonnées globalement
        if metadata:
            self.function_metadata[func.name] = metadata
        
        return metadata
    
    def call_transaction(self, trans: 'TransactionDecl', positional_args: List[Any], named_args: dict = None) -> Any:
        """
        Appelle une transaction avec audit logging automatique.
        
        Une transaction enregistre automatiquement :
        - L'heure de début/fin
        - Les paramètres d'entrée
        - La valeur de retour
        - Les erreurs éventuelles
        - Un hash pour l'intégrité
        
        Args:
            trans: Déclaration de transaction
            positional_args: Arguments positionnels
            named_args: Arguments nommés
        
        Returns:
            Valeur de retour de la transaction
        """
        if named_args is None:
            named_args = {}
        
        from runtime.primitives import audit_log, now
        import json
        
        # Timestamp de début
        start_time = now()
        
        # Combiner arguments positionnels et nommés
        param_values = {}
        for i, arg in enumerate(positional_args):
            if i < len(trans.parameters):
                param_values[trans.parameters[i].name] = arg
        
        for name, value in named_args.items():
            param_names = [p.name for p in trans.parameters]
            if name not in param_names:
                raise ExecutionError(f"Transaction {trans.name} has no parameter named '{name}'")
            param_values[name] = value
        
        if len(param_values) != len(trans.parameters):
            raise ExecutionError(
                f"Transaction {trans.name} expects {len(trans.parameters)} arguments, got {len(param_values)}"
            )
        
        # Log de début de transaction
        try:
            params_serializable = {k: str(v) for k, v in param_values.items()}
            audit_log(f"transaction_start_{trans.name}", params_serializable)
        except:
            pass  # Ne pas échouer si audit_log ne peut pas sérialiser
        
        # Créer snapshot pour rollback potentiel
        snapshot = self.create_scope_snapshot(self.current_scope)
        
        # Créer scope pour la transaction
        trans_scope = Scope(parent=self.current_scope)
        for param in trans.parameters:
            trans_scope.define_var(param.name, param_values[param.name])
        
        prev_scope = self.current_scope
        self.current_scope = trans_scope
        
        try:
            # Exécuter le corps de la transaction
            self.exec_block(trans.body)
            result = None
            
        except ReturnValue as ret:
            result = ret.value
            
        except Exception as e:
            # Erreur : rollback automatique
            self.restore_scope_snapshot(prev_scope, snapshot)
            
            # Log de l'échec
            try:
                audit_log(f"transaction_failed_{trans.name}", {"error": str(e)})
            except:
                pass
            
            # Si un bloc rollback est défini, l'exécuter
            if trans.rollback_block:
                try:
                    self.exec_block(trans.rollback_block)
                except Exception as rollback_error:
                    raise ExecutionError(
                        f"Rollback failed in {trans.name}: {str(rollback_error)}"
                    ) from rollback_error
            
            # Re-lever l'erreur
            raise ExecutionError(f"Transaction {trans.name} failed: {str(e)}") from e
        
        finally:
            self.current_scope = prev_scope
        
        # Transaction réussie : log de succès
        end_time = now()
        try:
            result_str = str(result) if result is not None else "None"
            audit_log(f"transaction_success_{trans.name}", {
                "result": result_str,
                "duration_ms": end_time - start_time
            })
        except:
            pass
        
        return result
    
    def create_scope_snapshot(self, scope: Scope) -> Dict[str, Any]:
        """
        Crée un snapshot (copie profonde) de l'état d'un scope.
        
        Args:
            scope: Scope à sauvegarder
        
        Returns:
            Dictionnaire contenant l'état des variables
        """
        import copy
        snapshot = {}
        
        # Copier les variables du scope actuel
        for name, value in scope.variables.items():
            # Copie profonde pour éviter les mutations
            try:
                snapshot[name] = copy.deepcopy(value)
            except:
                # Si deepcopy échoue (fonctions, etc.), copie shallow
                snapshot[name] = value
        
        return snapshot
    
    def restore_scope_snapshot(self, scope: Scope, snapshot: Dict[str, Any]):
        """
        Restaure l'état d'un scope depuis un snapshot.
        
        Args:
            scope: Scope à restaurer
            snapshot: Snapshot précédemment créé
        """
        # Supprimer les variables ajoutées depuis le snapshot
        vars_to_remove = set(scope.variables.keys()) - set(snapshot.keys())
        for var in vars_to_remove:
            if var in scope.variables:
                del scope.variables[var]
        
        # Restaurer les valeurs du snapshot
        for name, value in snapshot.items():
            scope.variables[name] = value
    
    # ============================================
    # Type Inference (Phase 3.1)
    # ============================================
    
    def infer_type(self, value: Any) -> str:
        """
        Infère le type d'une valeur.
        
        Args:
            value: Valeur dont on veut inférer le type
        
        Returns:
            Nom du type ('int', 'float', 'str', 'bool', 'Vec')
        """
        import numpy as np
        
        if isinstance(value, bool):
            # bool doit être testé avant int car bool est une sous-classe de int
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            return 'str'
        elif isinstance(value, np.ndarray):
            return 'Vec'
        else:
            # Type inconnu, retourner 'any' ou lever une erreur
            return 'any'
    
    # ============================================
    # Module System (Phase 3.2)
    # ============================================
    
    def load_module(self, module_name: str) -> Scope:
        """
        Charge un module NORMiL (.nor) ou Python
        
        Args:
            module_name: Nom du module à charger
        
        Returns:
            Scope contenant les exports du module
        
        Raises:
            ExecutionError: Si le module n'est pas trouvé
        """
        # Vérifier le cache
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        
        # Chercher d'abord un fichier .nor (modules NORMiL)
        module_file = None
        for module_path in self.module_paths:
            # Convertir en Path si c'est une string
            if isinstance(module_path, str):
                module_path = Path(module_path)
            
            candidate = module_path / f"{module_name}.nor"
            if candidate.exists():
                module_file = candidate
                break
        
        # Si fichier .nor trouvé, charger comme module NORMiL
        if module_file is not None:
            return self._load_normil_module(module_name, module_file)
        
        # Sinon, essayer de charger comme module Python (Phase 4.1)
        try:
            return self._load_python_module(module_name)
        except ImportError:
            raise ExecutionError(
                f"Module not found: {module_name}\n"
                f"Searched for .nor files in: {[str(p) for p in self.module_paths]}\n"
                f"Also tried to import as Python module but failed"
            )
    
    def _load_normil_module(self, module_name: str, module_file) -> Scope:
        """Charge un module NORMiL depuis un fichier .nor"""
        try:
            from parser.lexer import Lexer
            from parser.parser import Parser
            
            module_code = module_file.read_text(encoding='utf-8')
            lexer = Lexer(module_code)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            module_ast = parser.parse()
            
            # Créer un scope isolé pour le module
            module_scope = Scope()
            
            # Exécuter le module dans son propre scope
            prev_scope = self.current_scope
            self.current_scope = module_scope
            try:
                for stmt in module_ast.statements:
                    self.exec_statement(stmt)
            finally:
                self.current_scope = prev_scope
            
            # Mettre en cache
            self.loaded_modules[module_name] = module_scope
            
            return module_scope
            
        except FileNotFoundError:
            raise ExecutionError(f"Could not read module file: {module_file}")
        except Exception as e:
            raise ExecutionError(f"Error loading NORMiL module {module_name}: {str(e)}") from e
    
    def _load_python_module(self, module_name: str) -> Scope:
        """
        Charge un module Python natif (Phase 4.1)
        
        Args:
            module_name: Nom du module Python (ex: 'numpy', 'math', 'random')
        
        Returns:
            Scope wrappant le module Python
        """
        import importlib
        
        try:
            # Importer le module Python
            py_module = importlib.import_module(module_name)
            
            # Créer un scope qui wrappe le module Python
            module_scope = Scope()
            module_scope._python_module = py_module  # Garder référence au module Python
            
            # Mettre en cache
            self.loaded_modules[module_name] = module_scope
            
            return module_scope
            
        except ImportError as e:
            raise ImportError(f"Python module '{module_name}' not found") from e
    
    # ============================================
    # Statements
    # ============================================
    
    def exec_statement(self, stmt: Statement):
        """Exécute un statement"""
        
        # Import de module (Phase 3.2)
        if isinstance(stmt, ImportStmt):
            module_scope = self.load_module(stmt.module_name)
            module_name = stmt.alias if stmt.alias else stmt.module_name
            self.current_scope.define_module(module_name, module_scope)
            return
        
        # Déclaration de variable
        if isinstance(stmt, VarDecl):
            value = self.eval_expression(stmt.value)
            
            # Inférence de type si non spécifié (Phase 3.1)
            if stmt.var_type is None:
                inferred_type = self.infer_type(value)
                # Stocker le type inféré (optionnel, pour debugging)
                # On pourrait ajouter un attribut type_info au scope
            
            self.current_scope.define_var(stmt.name, value)
            return
        
        # Déclaration de fonction
        if isinstance(stmt, FunctionDecl):
            self.current_scope.define_function(stmt.name, stmt)
            return
        
        # Déclaration de transaction (Phase 6.4)
        if isinstance(stmt, TransactionDecl):
            self.current_scope.define_transaction(stmt.name, stmt)
            return
        
        # Déclaration de type
        if isinstance(stmt, TypeDecl):
            self.current_scope.define_type(stmt.name, stmt.type_def)
            return
        
        # Return
        if isinstance(stmt, ReturnStmt):
            value = self.eval_expression(stmt.value) if stmt.value else None
            raise ReturnValue(value)
        
        # If
        if isinstance(stmt, IfStmt):
            condition = self.eval_expression(stmt.condition)
            if condition:
                self.exec_block(stmt.then_block)
            elif stmt.else_block:
                self.exec_block(stmt.else_block)
            return
        
        # For
        if isinstance(stmt, ForStmt):
            iterable = self.eval_expression(stmt.iterable)
            for item in iterable:
                # Créer scope pour la boucle
                loop_scope = Scope(parent=self.current_scope)
                loop_scope.define_var(stmt.variable, item)
                
                prev_scope = self.current_scope
                self.current_scope = loop_scope
                try:
                    self.exec_block(stmt.body)
                finally:
                    self.current_scope = prev_scope
            return
        
        # While
        if isinstance(stmt, WhileStmt):
            while self.eval_expression(stmt.condition):
                self.exec_block(stmt.body)
            return
        
        # Match (basique)
        if isinstance(stmt, MatchStmt):
            value = self.eval_expression(stmt.value)
            for case in stmt.cases:
                if self.match_pattern(case.pattern, value):
                    # Vérifier condition where
                    if case.condition:
                        if not self.eval_expression(case.condition):
                            continue
                    self.exec_block(case.body)
                    break
            return
        
        # Expression statement
        if isinstance(stmt, ExpressionStmt):
            self.eval_expression(stmt.expression)
            return
        
        raise ExecutionError(f"Unknown statement type: {type(stmt)}")
    
    def exec_block(self, block: Block):
        """Exécute un bloc de statements"""
        for stmt in block.statements:
            self.exec_statement(stmt)
    
    def match_pattern(self, pattern: Pattern, value: Any) -> bool:
        """
        Vérifie si une valeur correspond à un pattern.
        
        Supporte:
        - Wildcard: _
        - Literal: 42, "hello", true
        - Type extraction: int(x), float(f), str(s), bool(b)
        - Struct extraction: EpisodicRecord(e)
        - Simple binding: x
        
        Args:
            pattern: Pattern à matcher
            value: Valeur à tester
        
        Returns:
            True si le pattern match, False sinon
        """
        
        # Wildcard: toujours match
        if isinstance(pattern, WildcardPattern):
            return True
        
        # Littéral: égalité stricte
        if isinstance(pattern, LiteralPattern):
            return pattern.value == value
        
        # Identificateur avec extraction de type ou binding simple
        if isinstance(pattern, IdentifierPattern):
            if pattern.inner_name:
                # Type extraction: case int(x), case float(f), etc.
                type_name = pattern.name
                var_name = pattern.inner_name
                
                # Vérifier le type et binder la variable
                if type_name == "int":
                    if isinstance(value, int):
                        self.current_scope.define_var(var_name, value)
                        return True
                    return False
                
                elif type_name == "float":
                    if isinstance(value, float):
                        self.current_scope.define_var(var_name, value)
                        return True
                    return False
                
                elif type_name == "str":
                    if isinstance(value, str):
                        self.current_scope.define_var(var_name, value)
                        return True
                    return False
                
                elif type_name == "bool":
                    if isinstance(value, bool):
                        self.current_scope.define_var(var_name, value)
                        return True
                    return False
                
                elif type_name == "EpisodicRecord":
                    if isinstance(value, EpisodicRecord):
                        self.current_scope.define_var(var_name, value)
                        return True
                    return False
                
                elif type_name == "Concept":
                    if isinstance(value, Concept):
                        self.current_scope.define_var(var_name, value)
                        return True
                    return False
                
                elif type_name == "Vec":
                    if isinstance(value, Vec):
                        self.current_scope.define_var(var_name, value)
                        return True
                    return False
                
                elif type_name == "list":
                    if isinstance(value, list):
                        self.current_scope.define_var(var_name, value)
                        return True
                    return False
                
                else:
                    # Type inconnu, on bind quand même (polymorphisme)
                    self.current_scope.define_var(var_name, value)
                    return True
            
            else:
                # Simple binding: case x (toujours match)
                self.current_scope.define_var(pattern.name, value)
                return True
        
        return False
    
    # ============================================
    # Exécution du programme
    # ============================================
    
    def execute(self, program: Program):
        """Exécute un programme NORMiL complet"""
        for stmt in program.statements:
            self.exec_statement(stmt)
        
        # Appeler main() si elle existe
        main_func = self.current_scope.get_function("main")
        if main_func:
            self.call_user_function(main_func, [])


# ============================================
# Helper pour exécuter du code NORMiL
# ============================================

def run_normil_code(code: str) -> Any:
    """
    Helper pour exécuter du code NORMiL.
    
    Args:
        code: Code source NORMiL
    
    Returns:
        Résultat de l'exécution
    """
    from parser.lexer import Lexer
    from parser.parser import Parser
    
    # Lexer
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    
    # Parser
    parser = Parser(tokens)
    ast = parser.parse()
    
    # Executor
    executor = Executor()
    executor.execute(ast)
    
    return executor


# ============================================
# Tests
# ============================================

if __name__ == '__main__':
    print("=== Test NORMiL Executor ===\n")
    
    # Test 1: Variable simple
    print("1. Variable:")
    code1 = """
    let x: int = 42
    let y: int = 10
    let z = x + y
    print(z)
    """
    exec1 = run_normil_code(code1)
    print()
    
    # Test 2: Fonction simple
    print("2. Fonction:")
    code2 = """
    fn add(a: int, b: int) -> int {
        return a + b
    }
    
    let result = add(10, 32)
    print(result)
    """
    exec2 = run_normil_code(code2)
    print()
    
    # Test 3: Opérateurs vectoriels
    print("3. Vecteurs:")
    code3 = """
    let v1 = zeros(128)
    let v2 = random(128, mean: 0.0, std: 1.0)
    let v3 = add(v1, v2)
    let similarity = dot(v2, v2)
    print(similarity)
    """
    exec3 = run_normil_code(code3)
    print()
    
    # Test 4: Boucle
    print("4. Boucle for:")
    code4 = """
    let numbers = [1, 2, 3, 4, 5]
    for n in numbers {
        print(n)
    }
    """
    exec4 = run_normil_code(code4)
    print()
    
    # Test 5: Condition
    print("5. If/Else:")
    code5 = """
    let x = 42
    if x > 40 {
        print("Grand")
    } else {
        print("Petit")
    }
    """
    exec5 = run_normil_code(code5)
    print()
    
    print("✅ Executor tests passed!")
