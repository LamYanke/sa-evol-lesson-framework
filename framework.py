import os
import math
import random
from entities import CodeIndividual, LessonBank, TribeAgent, MergerAgent
from components import LLMClient, IncrementalProfiler, CFGAnalyzer, TribeSelector


class SA_Evol_Lesson_Framework:
    def __init__(self, original_code, max_iterations=5):
        # 依赖注入
        self.llm = LLMClient(api_key=os.getenv("OPENAI_API_KEY"))
        self.profiler = IncrementalProfiler()
        self.cfg_tool = CFGAnalyzer()
        self.lesson_bank = LessonBank()  # LessonBank 2.0
        self.merger = MergerAgent(self.llm)
        self.selector = TribeSelector()

        # 部落初始化
        self.tribes = [
            TribeAgent("Architect", self.llm, self.cfg_tool),
            TribeAgent("Engineer", self.llm, self.cfg_tool),
            TribeAgent("Refiner", self.llm, self.cfg_tool)
        ]

        # 状态
        self.original_code = original_code
        self.latest_failed_code = None

        # 初始种群
        self.best_solution = CodeIndividual(original_code)
        self.best_solution.cfg_embedding = self.cfg_tool.extract_cfg_embedding(original_code)
        self.best_solution.speedup = 1.0

        self.max_iterations = max_iterations
        self.current_temp = 1.0

    def run(self):
        population = [self.best_solution]
        print(f"🚀 Framework 2.0 Start. Target: Maximize Speedup.")

        for iteration in range(self.max_iterations):
            print(f"\n>>> Iteration {iteration + 1}/{self.max_iterations} (Temp={self.current_temp:.2f})")

            # 1. 分析 (Profiling)
            profiler_data = self.profiler.profile_hotspots(self.best_solution.source_code)
            candidates = []

            # 2. 进化 (Evolution)
            for parent in population:
                # 激活部落
                active_tribes = self.selector.select(parent.source_code, self.tribes)

                for tribe in active_tribes:
                    # [Hybrid Retrieval] 混合检索
                    lesson_ctx = self.lesson_bank.retrieve(
                        target_code=parent.source_code,
                        target_cfg=parent.cfg_embedding,
                        requestor_role=tribe.role
                    )

                    # [Mutate] 变异 (带 Anchor & Anti-Pattern)
                    new_inds = tribe.mutate(
                        parent, lesson_ctx, profiler_data,
                        original_code=self.original_code,
                        failed_code=self.latest_failed_code
                    )
                    candidates.extend(new_inds)

            # 3. 评估与沉淀 (Eval & Deposit)
            valid_candidates = []
            for cand in candidates:
                success, speedup, err = self.profiler.full_evaluation(self.original_code, cand.source_code)

                if success:
                    cand.speedup = speedup
                    cand.is_valid = True
                    # 能量函数: 速度越快能量越低 (用于SA)
                    delta_e = math.log(speedup + 1e-5)

                    # [Delta Learning] 差量学习
                    if speedup > self.best_solution.speedup:
                        print(f"  [Improvement] {cand.origin_tribe}: {speedup:.2f}x")
                        self.lesson_bank.deposit_improvement(cand, self.best_solution, self.llm)
                    else:
                        self.lesson_bank.deposit_neutral(cand, "No gain", self.llm)

                    valid_candidates.append(cand)
                else:
                    # [Negative Lesson] 失败学习
                    print(f"  [Fail] {cand.origin_tribe}: {err}")
                    self.latest_failed_code = cand.source_code
                    self.lesson_bank.deposit_failure(cand, err, self.llm)

            # 4. 融合 (Merger)
            if len(valid_candidates) >= 2:
                valid_candidates.sort(key=lambda x: x.speedup, reverse=True)
                merged = self.merger.merge_weighted(valid_candidates[:3], original_code=self.original_code)
                if merged:
                    s, sp, _ = self.profiler.full_evaluation(self.original_code, merged.source_code)
                    if s:
                        merged.speedup = sp
                        merged.cfg_embedding = self.cfg_tool.extract_cfg_embedding(merged.source_code)
                        print(f"  [Merger] Combined success! Speedup: {sp:.2f}x")
                        valid_candidates.append(merged)

            # 5. 选择 (Selection - Simulated Annealing)
            valid_candidates.sort(key=lambda x: x.speedup, reverse=True)
            if not valid_candidates: continue

            # 更新全局最优
            if valid_candidates[0].speedup > self.best_solution.speedup:
                self.best_solution = valid_candidates[0]
                print(f"  ★ New Global Best: {self.best_solution.speedup:.2f}x")

            # 概率接受
            next_pop = [self.best_solution]
            for c in valid_candidates:
                if c == self.best_solution: continue
                # 简单退火逻辑：接受更差解的概率随温度降低
                delta = c.speedup - self.best_solution.speedup  # 负数
                prob = math.exp(delta / self.current_temp) if delta < 0 else 1.0

                if random.random() < prob:
                    next_pop.append(c)

            population = next_pop[:3]  # 保持种群规模
            self.current_temp *= 0.8

        return self.best_solution


