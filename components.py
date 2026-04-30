import ast
import re
import random
import concurrent.futures


# ==========================================
# 1. LLM 客户端
# ==========================================
class LLMClient:
    def __init__(self, api_key=None, base_url=None, model="gpt-4o"):
        self.client = None
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            except:
                pass
        self.model = model

    def generate(self, sys_p, usr_p):
        if not self.client: return self._mock(sys_p)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                temperature=0.7
            )
            return self._clean(resp.choices[0].message.content)
        except Exception as e:
            print(f"[LLM Error] {e}")
            return self._mock(sys_p)

    def _clean(self, text):
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def _mock(self, sys_p):
        return "def optimized(arr): return sum(arr) # Mock"


# ==========================================
# 2. CFG 工具 (特征提取)
# ==========================================
class CFGAnalyzer:
    def extract_cfg_embedding(self, code):
        try:
            tree = ast.parse(code)
            loops = sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))
            branches = sum(isinstance(n, ast.If) for n in ast.walk(tree))
            return [float(len(code)), float(loops), float(branches)]
        except:
            return [0.0, 0.0, 0.0]


# ==========================================
# 3. 确定性 Profiler (AST 插桩)
# ==========================================
class InstructionCounter(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        # 插入 __cnt[0] += 1
        incr = ast.Expr(value=ast.AugAssign(
            target=ast.Subscript(value=ast.Name(id='__cnt', ctx=ast.Load()), slice=ast.Constant(value=0),
                                 ctx=ast.Store()),
            op=ast.Add(), value=ast.Constant(value=1)))
        node.body.insert(0, incr)
        return node


class IncrementalProfiler:
    def __init__(self):
        # 模拟测试用例
        self.inputs = [[random.randint(0, 100) for _ in range(50)] for _ in range(5)]

    def full_evaluation(self, oracle_code, target_code):
        try:
            # 编译
            f_old, ctx_old = self._compile(oracle_code, "_old")
            f_new, ctx_new = self._compile(target_code, "_new")

            # 冒烟测试
            if f_old([1, 2, 3]) != f_new([1, 2, 3]):
                return False, 0.0, "Logic Mismatch"

            # 步数统计
            steps_old, steps_new = 0, 0
            for inp in self.inputs:
                ctx_old['__cnt'][0] = 0;
                f_old(list(inp));
                steps_old += ctx_old['__cnt'][0]
                ctx_new['__cnt'][0] = 0;
                f_new(list(inp));
                steps_new += ctx_new['__cnt'][0]

            speedup = (steps_old + 1e-5) / (steps_new + 1e-5)
            return True, speedup, ""
        except Exception as e:
            return False, 0.0, str(e)

    def _compile(self, code, suffix):
        tree = ast.parse(code)
        InstructionCounter().visit(tree)
        ast.fix_missing_locations(tree)

        # 重命名函数防止冲突
        target_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        target_fn.name += suffix

        ctx = {'__cnt': [0]};
        ctx.update(__builtins__)
        exec(compile(tree, "<string>", "exec"), ctx)
        return ctx[target_fn.name], ctx

    def profile_hotspots(self, code):
        # 简单静态分析
        return {"hotspots": ["inner_loop" if "for" in code else "main"]}


class TribeSelector:
    def select(self, code, tribes):
        return tribes  # 默认激活所有部落