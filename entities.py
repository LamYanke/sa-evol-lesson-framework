import hashlib
import ast
import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict

# 尝试导入 BM25，若无则降级处理
try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("[Warn] rank_bm25 not found. Falling back to simple retrieval.")


# ==========================================
# 1. 代码个体 (Individual)
# ==========================================
@dataclass
class CodeIndividual:
    source_code: str
    cfg_embedding: List[float] = field(default_factory=list)  # [Length, Loops, Branches]
    speedup: float = 0.0
    is_valid: bool = False
    energy: float = 0.0
    origin_tribe: str = "Init"
    generation: int = 0
    _id: str = field(init=False)

    def __post_init__(self):
        # 计算唯一 ID
        clean_code = "".join(self.source_code.split())
        self._id = hashlib.md5(clean_code.encode('utf-8')).hexdigest()


# ==========================================
# 2. 经验银行 (LessonBank 2.0)
# ==========================================
@dataclass
class Lesson:
    content: str
    impact_score: float  # 成功=Speedup, 失败=5.0
    lesson_type: str  # "Tactic", "Negative", "Ineffective"
    origin_tribe: str  # 来源角色

    # Context Storage (存储"案发现场"的特征)
    pre_code: str  # 父代代码 (用于展示)
    pre_ast_dump: str  # 父代 AST String (用于 BM25 / 去重)
    pre_cfg: List[float]  # 父代 CFG 向量 (用于结构距离)
    post_code: Optional[str] = None  # 成功后的代码 (参考答案)


class LessonBank:
    def __init__(self):
        self.lessons: List[Lesson] = []
        self.seen_signatures: Set[str] = set()  # AST指纹去重集合
        self.bm25_index = None
        self.dirty = True  # 索引是否需要更新
        self._seed_universal_lessons()

    def _seed_universal_lessons(self):
        """预埋通用规则"""
        seeds = [
            ("Avoid nested loops O(N^2); use HashMaps/Sets for O(N).", "Architect"),
            ("Unroll tight loops to reduce interpreter overhead.", "Engineer"),
            ("Use built-in functions (sum, max) instead of manual loops.", "Refiner")
        ]
        for c, t in seeds:
            # 种子经验使用特殊标记
            l = Lesson(c, 1.5, "Tactic", t, "", "universal_seed", [], None)
            self.lessons.append(l)

    def _get_ast_dump(self, code):
        """获取 AST 结构字符串，忽略格式差异"""
        if not code: return ""
        try:
            return ast.dump(ast.parse(code))
        except:
            return "".join(code.split())

    def _calculate_signature(self, lesson_type, pre_code, post_code):
        """计算经验指纹: Type + Pre_AST_Hash + Post_AST_Hash"""
        pre_hash = hashlib.md5(self._get_ast_dump(pre_code).encode()).hexdigest()
        post_hash = "None"
        if post_code:
            post_hash = hashlib.md5(self._get_ast_dump(post_code).encode()).hexdigest()
        return f"{lesson_type}:{pre_hash}->{post_hash}"

    def _update_index(self):
        """懒加载更新 BM25 索引"""
        if not HAS_BM25 or not self.dirty: return
        if self.lessons:
            # 仅索引非种子的有效 AST
            corpus = [l.pre_ast_dump.split() for l in self.lessons if l.pre_ast_dump != "universal_seed"]
            if corpus:
                self.bm25_index = BM25Okapi(corpus)
            self.dirty = False

    # --- 存入逻辑 (Write) ---

    def deposit_improvement(self, child, parent, llm_client):
        """存入成功战术 (Delta Learning)"""
        if child.speedup <= 1.05:
            self.deposit_neutral(child, "Low gain", llm_client)
            return

        # 1. 去重检查
        sig = self._calculate_signature("Tactic", parent.source_code, child.source_code)
        if sig in self.seen_signatures: return

        # 2. 生成战术总结
        prompt = f"""
        Analyze the specific change from Parent to Child that caused {child.speedup:.2f}x speedup.
        Parent Snippet: \n{parent.source_code[:400]}...
        Child Snippet: \n{child.source_code[:400]}...
        Output one concise optimization tactic (e.g., "Replaced list lookup with set").
        """
        content = llm_client.generate("Lesson Learner", prompt)

        # 3. 存入 (注意：存的是 Parent 的特征)
        self.lessons.append(Lesson(
            content=f"[SUCCESS TACTIC] {content}",
            impact_score=child.speedup,
            lesson_type="Tactic",
            origin_tribe=child.origin_tribe,
            pre_code=parent.source_code,
            pre_ast_dump=self._get_ast_dump(parent.source_code),
            pre_cfg=parent.cfg_embedding,
            post_code=child.source_code
        ))
        self.seen_signatures.add(sig)
        self.dirty = True

    def deposit_failure(self, failed_code, error_msg, llm_client):
        """存入失败教训 (Negative)"""
        sig = self._calculate_signature("Negative", failed_code.source_code, None)
        if sig in self.seen_signatures: return

        prompt = f"Code failed:\n{error_msg}\nExplain what to avoid in one sentence."
        content = llm_client.generate("Debugger", prompt)

        self.lessons.append(Lesson(
            content=f"[AVOID] {content}",
            impact_score=5.0,  # 高权重
            lesson_type="Negative",
            origin_tribe=failed_code.origin_tribe,
            pre_code=failed_code.source_code,  # 存的是导致报错的代码
            pre_ast_dump=self._get_ast_dump(failed_code.source_code),
            pre_cfg=failed_code.cfg_embedding,
            post_code=None
        ))
        self.seen_signatures.add(sig)
        self.dirty = True

    def deposit_neutral(self, code, reason, llm_client):
        """存入无效尝试"""
        sig = self._calculate_signature("Ineffective", code.source_code, None)
        if sig in self.seen_signatures: return

        self.lessons.append(Lesson(
            content="Attempted optimization strategy yielded no speedup.",
            impact_score=1.2,
            lesson_type="Ineffective",
            origin_tribe=code.origin_tribe,
            pre_code=code.source_code,
            pre_ast_dump=self._get_ast_dump(code.source_code),
            pre_cfg=code.cfg_embedding,
            post_code=None
        ))
        self.seen_signatures.add(sig)
        self.dirty = True

    # --- 读取逻辑 (Read / Retrieval) ---

    def retrieve(self, target_code, target_cfg, requestor_role, top_k=3) -> str:
        if not self.lessons: return ""
        self._update_index()

        # 1. BM25 文本/结构相似度
        bm25_scores = [0.0] * len(self.lessons)
        if HAS_BM25 and self.bm25_index:
            try:
                target_ast = self._get_ast_dump(target_code)
                bm25_scores = self.bm25_index.get_scores(target_ast.split())
            except:
                pass

        # 2. 定义角色关注权重 [Length, Loops, Branches]
        cfg_weights = {
            "Architect": [0.1, 5.0, 1.0],  # 严查循环
            "Engineer": [0.5, 3.0, 3.0],  # 严查热点结构
            "Refiner": [2.0, 0.5, 2.0],  # 关注复杂度
        }.get(requestor_role, [1.0, 1.0, 1.0])

        scored = []
        for i, l in enumerate(self.lessons):
            # A. 加权 CFG 距离
            w_dist = 10.0
            if l.pre_cfg and target_cfg:
                diff_sq = sum(cfg_weights[d] * ((l.pre_cfg[d] - target_cfg[d]) ** 2) for d in range(3))
                w_dist = math.sqrt(diff_sq)
            elif l.pre_ast_dump == "universal_seed":
                w_dist = 0.5  # 种子距离适中

            # B. 角色加成 (Role Bonus)
            role_bonus = 0.5
            if l.lesson_type == "Negative":
                role_bonus = 3.0  # 报错全员可见
            elif l.origin_tribe == requestor_role:
                role_bonus = 2.0  # 同行经验
            elif l.origin_tribe == "Universal":
                role_bonus = 1.0

            # C. 综合打分公式
            text_sim = bm25_scores[i] + 1.0
            score = (text_sim * l.impact_score * role_bonus) / (w_dist + 0.1)
            scored.append((score, l))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 3. 内容语义去重 (防止 Top-3 都是同一条建议)
        final_results = []
        seen_content_hashes = set()

        for s, l in scored:
            if len(final_results) >= top_k: break
            if s < 0.5: continue

            # 简易内容去重
            content_hash = hashlib.md5("".join(l.content.split()).encode()).hexdigest()
            if content_hash in seen_content_hashes: continue
            seen_content_hashes.add(content_hash)

            # 格式化
            tag = "✅ Ref" if l.lesson_type != "Negative" else "❌ Avoid"
            msg = f"- [{tag} from {l.origin_tribe}] {l.content}"
            if l.pre_code:
                snip = l.pre_code[:60].replace('\n', ' ') + "..."
                msg += f" (Context: {snip})"
            final_results.append(msg)

        return "\n".join(final_results)


# ==========================================
# 3. 部落代理 (TribeAgent 2.0)
# ==========================================
class TribeAgent:
    def __init__(self, role, llm, cfg_tool):
        self.role = role
        self.llm = llm
        self.cfg_tool = cfg_tool

    def mutate(self, parent, lesson_context, profiler_data, original_code=None, failed_code=None):
        # 1. 构造上下文
        anchor = f"=== [ANCHOR] Original Logic (MUST KEEP) ===\n{original_code}\n" if original_code else ""
        anti = ""
        if failed_code:
            anti = f"=== [ANTI-PATTERN] Failed Code (DO NOT COPY) ===\n{failed_code[:500]}...\n"

        benchmark = f"{parent.speedup:.1f}x" if parent.speedup > 0 else "Baseline"

        # 2. 角色特定 Prompt
        sys_prompt = f"You are a Code {self.role}. Optimize the [TARGET] code."
        user_prompt = ""

        if self.role == "Architect":
            user_prompt = f"""
            MISSION: BREAK THE LIMIT (Current: {benchmark}).
            Role: Algorithm Architect. 
            Focus: Big-O Complexity, Data Structures (Maps/Sets).
            Strategy: Replace nested loops O(N^2) with O(N). Change algorithm paradigm.

            LESSONS (Weighted for Structure):
            {lesson_context}

            {anchor}
            {anti}
            === [TARGET] Current Version ===
            {parent.source_code}
            """
        elif self.role == "Engineer":
            hotspots = ", ".join(profiler_data.get("hotspots", ["whole_function"]))
            user_prompt = f"""
            MISSION: BREAK THE LIMIT (Current: {benchmark}).
            Role: Performance Engineer.
            Focus: Runtime Hotspots [{hotspots}].
            Strategy: Unroll loops, vectorization, cache locality. Do NOT change Algo Structure.

            LESSONS (Weighted for Hotspots):
            {lesson_context}

            {anchor}
            {anti}
            === [TARGET] Current Version ===
            {parent.source_code}
            """
        elif self.role == "Refiner":
            user_prompt = f"""
            MISSION: POLISH CODE.
            Role: Code Refiner.
            Focus: Pythonic Idioms, Readability.
            Strategy: List comprehensions, clean naming. Maintain logic.

            LESSONS (Weighted for Style):
            {lesson_context}

            {anchor}
            === [TARGET] Current Version ===
            {parent.source_code}
            """

        # 3. 生成代码
        code = self.llm.generate(sys_prompt, user_prompt)

        ind = CodeIndividual(source_code=code, origin_tribe=self.role, generation=parent.generation + 1)
        ind.cfg_embedding = self.cfg_tool.extract_cfg_embedding(code)
        return [ind]  # 返回列表以兼容接口


# ==========================================
# 4. 合并代理
# ==========================================
import difflib


class MergerAgent:
    def __init__(self, llm):
        self.role = "Merger"
        self.llm = llm

    def _get_diff_summary(self, original, modified, label):
        """生成带标签的差异摘要"""
        diff = difflib.unified_diff(
            original.splitlines(),
            modified.splitlines(),
            fromfile='Base',
            tofile=label,
            n=1,  # 减少上下文行数以节省 Token
            lineterm=''
        )
        return "\n".join(list(diff))

    def merge_weighted(self, ranked_candidates, original_code=None):
        """
        多源融合策略：
        1. 选定 Leader (Rank 1) 作为基底。
        2. 将 Rank 2, Rank 3... (Followers) 视为 '功能补丁'。
        3. 让 LLM 将所有补丁的意图应用到 Leader 上。
        """
        if len(ranked_candidates) < 2 or original_code is None:
            return None

        # 1. 确定基底 (Leader) 和 贡献者 (Followers)
        # 我们可以放宽到 Top 3，防止上下文过长
        top_k = ranked_candidates[:3]
        leader = top_k[0]
        followers = top_k[1:]

        # 2. 构造所有 Follower 的补丁信息
        patches_text = ""
        for i, cand in enumerate(followers):
            # 计算相对于原始代码(original_code)的改动，这样 LLM 能看懂它的意图
            # 注意：这里对比 original 而不是 leader，是因为我们需要提取由于"不同策略"产生的纯粹意图
            diff = self._get_diff_summary(original_code, cand.source_code, cand.origin_tribe)
            if diff.strip():
                patches_text += f"\n=== [Patch {i + 1} Source: {cand.origin_tribe} (Speedup: {cand.speedup:.2f}x)] ===\n{diff[:1500]}\n"

        if not patches_text:
            return None

        # 3. 构造聚合 Prompt
        prompt = f"""
        Act as a Lead Code Integrator. You have a "Base Code" that is currently the fastest.
        You also have "Patches" from other agents that attempted different optimizations (e.g., style, loop unrolling, etc.).

        [MISSION]
        Combine the valid logic from the [Patches] into the [Base Code] to create a "Super Version".

        RULES:
        1. **Primary Structure**: MUST follow the [Base Code] logic (it is the fastest).
        2. **Absorption**: Look at the [Patches]. If a patch adds type hints, better naming, or small local optimizations, APPLY them to the Base Code.
        3. **Conflict Resolution**: If a Patch conflicts with the Base Code's core algorithm, IGNORE the Patch.

        === [Base Code] (Leader) ===
        {leader.source_code}

        {patches_text}

        Output the merged Python code only.
        """

        code = self.llm.generate("Lead Integrator", prompt)

        return CodeIndividual(
            source_code=code,
            origin_tribe="Merger",
            generation=leader.generation + 1
        )