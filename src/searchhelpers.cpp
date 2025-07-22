#include "searchhelpers.hpp"
#include "config.hpp"
#include "mcts.hpp"
#include <algorithm>
#include <cmath>

namespace SearchHelpers {

    float calculateFPUValue(
        const SearchStats& parentStats,
        float policyProbMassVisited,
        StoneColor currentPlayer,
        const FPUParams& params,
        bool isRoot
    ) {
        if (parentStats.totalVisits <= 0) {
            // 如果父节点没有访问，返回中性值
            return 0.0f;
        }

        // 使用父节点的平均效用作为基准
        float parentUtilityForFPU = parentStats.utilityAvg;
        
        // 计算 FPU 削减量
        float fpuReductionMax = isRoot ? params.fpuReductionMax * 1.5f : params.fpuReductionMax;
        float fpuLossProp = isRoot ? params.fpuLossProp : params.fpuLossProp;
        
        // 基于已访问策略质量的削减
        float reduction = fpuReductionMax * std::sqrt(policyProbMassVisited);
        
        float fpuValue;
        if (currentPlayer == Config::WHITE_STONE) {
            fpuValue = parentUtilityForFPU - reduction;
        } else {
            fpuValue = parentUtilityForFPU + reduction;
        }
        
        // 向损失方向倾斜
        if (fpuLossProp > 0.0f) {
            float lossValue = (currentPlayer == Config::WHITE_STONE) ? -1.0f : 1.0f;
            fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
        }
        
        return fpuValue;
    }

    float calculateExploreScaling(
        float totalChildWeight,
        float parentUtilityStdevFactor,
        const FPUParams& params
    ) {
        // 计算动态探索系数
        float cpuctExploration = params.cpuctExploration;
        if (params.cpuctExplorationLog > 0.0f) {
            cpuctExploration += params.cpuctExplorationLog * 
                std::log((totalChildWeight + params.cpuctExplorationBase) / params.cpuctExplorationBase);
        }
        
        // 微小偏移量，防止除零
        constexpr float WEIGHT_OFFSET = 0.01f;
        
        return cpuctExploration * 
               std::sqrt(totalChildWeight + WEIGHT_OFFSET) * 
               parentUtilityStdevFactor;
    }

    float calculateExploreSelectionValue(
        float exploreScaling,
        float nnPolicyProb,
        float childWeight,
        float childUtility,
        StoneColor currentPlayer
    ) {
        if (nnPolicyProb < 0) {
            return -1e9f; // 非法移动的极低价值
        }

        // 探索组件：类似 UCB 的探索项
        float exploreComponent = exploreScaling * nnPolicyProb / (1.0f + childWeight);

        // 价值组件：从当前玩家视角调整
        float valueComponent = (currentPlayer == Config::WHITE_STONE) ? childUtility : -childUtility;
        
        return exploreComponent + valueComponent;
    }

    float calculateNewChildSelectionValue(
        float exploreScaling,
        float nnPolicyProb,
        float fpuValue
    ) {
        if (nnPolicyProb < 0) {
            return -1e9f; // 非法移动
        }

        // 新子节点的权重为 0
        float childWeight = 0.0f;
        float exploreComponent = exploreScaling * nnPolicyProb / (1.0f + childWeight);
        
        return exploreComponent + fpuValue;
    }

    void updateSearchStats(SearchStats& stats, const Node& node) {
        stats.totalVisits = node.visit_count;
        
        if (node.visit_count > 0) {
            stats.utilityAvg = node.value_sum / static_cast<float>(node.visit_count);
            stats.weightSum = static_cast<float>(node.visit_count); // 简化：权重 = 访问次数
            
            // 改进的标准差计算：先计算观察值，然后结合先验
            float observedStdev = std::max(0.02f, std::abs(stats.utilityAvg) * 0.1f);
            
            // 使用先验改进标准差估计
            stats.utilityStdev = calculateUtilityStdev(
                observedStdev,
                Config::FPU::UTILITY_STDEV_PRIOR,
                Config::FPU::UTILITY_STDEV_PRIOR_WEIGHT,
                node.visit_count
            );
        } else {
            stats.utilityAvg = 0.0f;
            stats.weightSum = 0.0f;
            // 无访问时使用先验标准差
            stats.utilityStdev = Config::FPU::UTILITY_STDEV_PRIOR;
        }
    }

    FPUParams getAdaptiveFPUParams(const SearchStats& stats, bool isRoot) {
        FPUParams params;
        
        // 基础参数
        params.cpuctExploration = Config::C_PUCT;
        params.fpuReductionMax = Config::FPU::REDUCTION_MAX;
        params.fpuLossProp = Config::FPU::LOSS_PROP;
        params.utilityStdevScale = Config::FPU::UTILITY_STDEV_SCALE;
        
        // 第一优先级：KataGo高级参数
        params.valueWeightExponent = Config::FPU::VALUE_WEIGHT_EXPONENT;
        params.uncertaintyCoeff = Config::FPU::UNCERTAINTY_COEFF;
        params.utilityStdevPrior = Config::FPU::UTILITY_STDEV_PRIOR;
        params.utilityStdevPriorWeight = Config::FPU::UTILITY_STDEV_PRIOR_WEIGHT;
        params.useUncertaintyWeighting = Config::FPU::USE_UNCERTAINTY_WEIGHTING;
        
        // 根据游戏阶段调整参数
        if (stats.totalVisits < 50) {
            // 早期游戏：更多探索
            params.fpuReductionMax *= 0.8f;
            params.cpuctExploration *= 1.2f;
            // 早期游戏不确定性更高，增强不确定性系数
            params.uncertaintyCoeff *= 1.3f;
        } else if (stats.totalVisits > 200) {
            // 后期游戏：更多利用
            params.fpuReductionMax *= 1.2f;
            params.cpuctExploration *= 0.9f;
            // 后期降低不确定性系数，更信任评估
            params.uncertaintyCoeff *= 0.8f;
        }
        
        // 根据局面不确定性调整 - 借鉴KataGo的uncertaintyCoeff思想
        if (stats.utilityStdev > 0.1f) {
            // 高不确定性：增加探索，类似KataGo的不确定性权重机制
            params.fpuReductionMax *= 0.9f;
            // 添加基于不确定性的CPUCT调整 (类似cpuctUtilityStdevScale)
            float uncertaintyBoost = 1.0f + params.utilityStdevScale * (stats.utilityStdev / 0.02f - 1.0f);
            params.cpuctExploration *= uncertaintyBoost;
        }

        // 根节点特殊处理
        if (isRoot) {
            params.fpuReductionMax *= params.rootFpuMultiplier;
            // 根节点使用更强的价值权重，快速淘汰差的变化
            params.valueWeightExponent *= 1.2f;
        }
        
        return params;
    }

    void calculateChildrenSelectionValues(
        const Node& parent,
        const std::vector<int>& legalMoves,
        std::vector<float>& selectionValues,
        const FPUParams& params
    ) {
        // 预计算通用信息
        SearchStats parentStats;
        updateSearchStats(parentStats, parent);
        
        float policyProbMassVisited = 0.0f;
        float totalChildWeight = 0.0f;
        
        for (int moveIdx : legalMoves) {
            if (parent.children[moveIdx] != nullptr) {
                policyProbMassVisited += parent.pi[moveIdx];
                
                // 应用不确定性权重到子节点权重计算
                float baseWeight = static_cast<float>(parent.children[moveIdx]->visit_count);
                if (params.useUncertaintyWeighting && parent.children[moveIdx]->visit_count > 0) {
                    float childUtility = -(parent.children[moveIdx]->value_sum / baseWeight);
                    float uncertainty = std::abs(childUtility - parentStats.utilityAvg);
                    float uncertaintyWeight = calculateUncertaintyWeight(uncertainty, params.uncertaintyCoeff);
                    baseWeight *= uncertaintyWeight;
                }
                totalChildWeight += baseWeight;
            }
        }
        
        float fpuValue = calculateFPUValue(
            parentStats, policyProbMassVisited, parent.current_color, params
        );
        
        float parentUtilityStdevFactor = 1.0f + params.utilityStdevScale * 
            (parentStats.utilityStdev / 0.02f - 1.0f);
        float exploreScaling = calculateExploreScaling(
            totalChildWeight, parentUtilityStdevFactor, params
        );
        
        // 批量计算选择价值
        selectionValues.resize(legalMoves.size());
        for (size_t i = 0; i < legalMoves.size(); ++i) {
            int moveIdx = legalMoves[i];
            
            if (parent.children[moveIdx] != nullptr) {
                const Node* child = parent.children[moveIdx].get();
                float childUtility = child->visit_count > 0 
                    ? -(child->value_sum / static_cast<float>(child->visit_count))
                    : fpuValue;
                
                float childWeight = static_cast<float>(child->visit_count);
                
                // 应用价值权重：对表现差的子节点降权
                if (params.valueWeightExponent > 0.0f && child->visit_count > 0) {
                    float valueWeight = calculateValueWeight(
                        parentStats.utilityAvg, childUtility, params.valueWeightExponent
                    );
                    childWeight *= valueWeight;
                }
                
                // 应用不确定性权重
                if (params.useUncertaintyWeighting && child->visit_count > 0) {
                    float uncertainty = std::abs(childUtility - parentStats.utilityAvg);
                    float uncertaintyWeight = calculateUncertaintyWeight(uncertainty, params.uncertaintyCoeff);
                    childWeight *= uncertaintyWeight;
                }
                
                selectionValues[i] = calculateExploreSelectionValue(
                    exploreScaling, parent.pi[moveIdx], childWeight, childUtility, parent.current_color
                );
            } else {
                selectionValues[i] = calculateNewChildSelectionValue(
                    exploreScaling, parent.pi[moveIdx], fpuValue
                );
            }
        }
    }

    // 第一优先级：KataGo高级功能实现
    
    float calculateValueWeight(
        float parentUtility,
        float childUtility,
        float valueWeightExponent
    ) {
        if (valueWeightExponent <= 0.0f) {
            return 1.0f; // 禁用价值权重时返回1
        }
        
        // 计算子节点相对于父节点的价值差异
        float utilityDiff = std::max(0.0f, parentUtility - childUtility);
        
        // 使用指数函数对价值差的子节点降权
        // 类似KataGo的valueWeightExponent机制
        return std::exp(-valueWeightExponent * utilityDiff);
    }

    float calculateUncertaintyWeight(
        float uncertainty,
        float uncertaintyCoeff
    ) {
        if (uncertaintyCoeff <= 0.0f || uncertainty <= 0.0f) {
            return 1.0f; // 禁用或无不确定性时返回1
        }
        
        // 不确定性越高，权重倍数越大（更多访问）
        // 类似KataGo的uncertaintyCoeff机制
        return 1.0f + uncertaintyCoeff * uncertainty;
    }

    float calculateUtilityStdev(
        float observedStdev,
        float prior,
        float priorWeight,
        int totalVisits
    ) {
        if (totalVisits <= 0) {
            return prior; // 无访问时返回先验
        }
        
        // 将先验和观察到的标准差进行加权平均
        // 访问次数越多，观察值权重越大
        float observedWeight = static_cast<float>(totalVisits);
        float totalWeight = priorWeight + observedWeight;
        
        return (prior * priorWeight + observedStdev * observedWeight) / totalWeight;
    }

} // namespace SearchHelpers
