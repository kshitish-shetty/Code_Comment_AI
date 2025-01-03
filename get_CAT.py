import javalang # type: ignore
import json

def parse_java_to_ast(java_code):
    """
    Parses Java code into an AST using javalang.
    Returns a list of nodes with type, value, and position.
    """
    try:
        tokens = javalang.tokenizer.tokenize(java_code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError) as e:
        print(f"Parsing error: {e}")
        return []

    ast_nodes = []
    for path, node in tree:
        node_data = {
            "type": type(node).__name__,
            "value": getattr(node, 'name', None),
            "position": getattr(node, 'position', None)
        }
        ast_nodes.append(node_data)
    return ast_nodes

def generate_code_aligned_type_sequence(java_code):
    """
    Generates a code-aligned type sequence from Java code.
    Maps tokens to their AST types.
    """
    try:
        tokens = list(javalang.tokenizer.tokenize(java_code))
        ast_nodes = parse_java_to_ast(java_code)
    except Exception as e:
        print(f"Error generating sequence: {e}")
        return []

    aligned_sequence = []
    for token in tokens:
        token_type = token.__class__.__name__
        aligned_type = next((node['type'] for node in ast_nodes if node['value'] == token.value), token_type)
        aligned_sequence.append(aligned_type)

    return " ".join(aligned_sequence)

# Example usage:
# java_code = """
# public boolean isValidUse(AnnotatedPrimitiveType type, Tree tree) {
#     return BOOL_;
# }
# """

# # Parse AST
# ast_nodes = parse_java_to_ast(java_code)
# # print("AST Nodes:", json.dumps(ast_nodes, indent=2))

# # Generate Code-Aligned Type Sequence
# aligned_sequence = generate_code_aligned_type_sequence(java_code)
# print("\nCode-Aligned Type Sequence:")
# # for token, aligned_type in aligned_sequence:
# #     print(f"{aligned_type} ", end=" ")

# print(aligned_sequence)
