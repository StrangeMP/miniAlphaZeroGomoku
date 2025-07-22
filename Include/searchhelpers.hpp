#pragma once
#include <vector>

// 前向声明避免循环依赖
struct Node;

// 使用 int 代替 STONE_COLOR 来避免循环依赖
using StoneColor = int;

namespace SearchHelpers {
    // FPU 参数结构
    struct FPUParams {
        float fpuReductionMax = 0.2f;           // 最大 FPU 削减量
        float fpuLossProp = 0.0f;               // 向损失方向倾斜的比例
        float cpuctExploration = 1.0f;          // 基础探索系数
        float cpuctExplorationLog = 0.0f;       // 对数探索系数
        float cpuctExplorationBase = 19652.0f;  // 探索基数
        bool useAdvancedFPU = true;             // 是否使用高级 FPU
        float rootFpuMultiplier = 1.5f;         // 根节点 FPU 倍数
        float utilityStdevScale = 0.1f;         // 效用标准差缩放
        
        // 第一优先级：KataGo高级参数
        float valueWeightExponent = 0.5f;       // 价值权重指数
        float uncertaintyCoeff = 0.15f;         // 不确定性系数
        float utilityStdevPrior = 0.25f;        // 效用标准差先验
        float utilityStdevPriorWeight = 1.0f;   // 先验权重
        bool useUncertaintyWeighting = true;    // 启用不确定性权重
    };

    // 搜索统计信息
    struct SearchStats {
        int totalVisits = 0;
        float utilityAvg = 0.0f;
        float utilityStdev = 0.02f;             // 效用标准差
        float weightSum = 0.0f;
    };

    // 核心 FPU 计算函数
    float calculateFPUValue(
        const SearchStats& parentStats,
        float policyProbMassVisited,
        StoneColor currentPlayer,
        const FPUParams& params,
        bool isRoot = false
    );

    // 探索缩放计算
    float calculateExploreScaling(
        float totalChildWeight,
        float parentUtilityStdevFactor,
        const FPUParams& params
    );

    // 子节点选择价值计算
    float calculateExploreSelectionValue(
        float exploreScaling,
        float nnPolicyProb,
        float childWeight,
        float childUtility,
        StoneColor currentPlayer
    );

    // 新子节点的选择价值计算
    float calculateNewChildSelectionValue(
        float exploreScaling,
        float nnPolicyProb,
        float fpuValue
    );

    // 从节点统计信息更新搜索统计
    void updateSearchStats(SearchStats& stats, const Node& node);

    // 高级特性：自适应 FPU 参数
    FPUParams getAdaptiveFPUParams(const SearchStats& stats, bool isRoot = false);

    // 批量计算子节点选择价值（性能优化）
    void calculateChildrenSelectionValues(
        const Node& parent,
        const std::vector<int>& legalMoves,
        std::vector<float>& selectionValues,
        const FPUParams& params
    );

    // 第一优先级：KataGo高级功能
    // 价值权重调整 - 对表现差的子节点降权
    float calculateValueWeight(
        float parentUtility,
        float childUtility,
        float valueWeightExponent
    );

    // 不确定性权重计算
    float calculateUncertaintyWeight(
        float uncertainty,
        float uncertaintyCoeff
    );

    // 改进的标准差计算，包含先验
    float calculateUtilityStdev(
        float observedStdev,
        float prior,
        float priorWeight,
        int totalVisits
    );
}
