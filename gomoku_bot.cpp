#include <iostream>
#include <vector>
#include <string> // 用于 std::string, std::getline, std::stoi
// #include <sstream> // 此头文件未被直接使用，可以考虑移除
#include <algorithm> // 用于 std::fill
#include <map>
#include <time.h> // 用于 srand
#include <stdlib.h> // 用于 rand
#include <cmath>   // 用于 sqrt, log, round
#include <chrono>  // 用于时间限制

// 根据用户提供的代码调整 JSON 库的包含
#ifdef _BOTZONE_ONLINE
#include "jsoncpp/json.h" // Botzone 环境下的路径
#else
#include <json/json.h>    // 本地环境的预期路径 (如果编译失败，请确保此路径有效或调整它)
#endif

const int SIZE = 15; // 棋盘大小
std::vector<std::vector<int>> Grid(SIZE, std::vector<int>(SIZE, -1)); // 全局棋盘状态, -1: 空, 0: 黑棋, 1: 白棋

// 从 Alpha-go/mainWork.cpp 引入的定义，已适配 gomoku_bot.cpp
#define SELECT_NUM 50      // MCTS选择阶段的迭代次数 (为满足1秒限制而调整)
#define STA_NUM 5          // MCTS每个扩展节点模拟的次数 (为满足1秒限制而调整)
#define MP(x,y) std::make_pair(x,y) // std::make_pair 的简写

const int SEARCH_RANGE = 2;   // MCTS扩展节点时的搜索范围半径
const double INF = 1e9 + 7;   // 定义一个较大的值，用于UCB计算中的未访问节点

// Alpha-go/mainWork.cpp 中的棋盘结构体
// 已适配为使用 -1 表示空，0 表示黑棋，1 表示白棋，与全局 Grid 一致
typedef struct ChessNode { // 重命名为 ChessNode 以避免潜在的类名冲突
    int g[SIZE][SIZE];

    ChessNode() { // 默认构造函数，初始化为空棋盘
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                g[i][j] = -1; // 初始化为 -1 (空)
            }
        }
    }
    // 从全局 Grid 复制数据的构造函数
    ChessNode(const std::vector<std::vector<int>>& globalGrid) {
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                this->g[i][j] = globalGrid[i][j];
            }
        }
    }
} ChessNode;

// ChessNode 的运算符重载
bool operator<(const ChessNode& x, const ChessNode& y) noexcept { // 用于 std::map 的键比较
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (x.g[i][j] < y.g[i][j]) return true;
            else if (x.g[i][j] > y.g[i][j]) return false;
        }
    }
    return false;
}

bool operator==(const ChessNode& x, const ChessNode& y) noexcept { // 用于判断棋盘状态是否相等
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (x.g[i][j] != y.g[i][j]) return false;
        }
    }
    return true;
}

// Alpha-go/mainWork.cpp 中的节点属性结构体
typedef struct Property { // 重命名为 Property
    double a; // UCB中的获胜分数 (或评估分数)
    double b; // UCB中的访问次数
    std::vector<ChessNode> vec; // 存储子节点的棋盘状态
} Property;

// Alpha-go/mainWork.cpp 中的全局变量，已适配
std::map<ChessNode, Property> mcts_mp; // MCTS节点信息表 (原 mp)
std::map<ChessNode, ChessNode> mcts_fa;   // MCTS父节点表 (原 fa)
std::pair<int, int> current_search_center; // 当前MCTS搜索的中心点 (原 center)

// MCTS 函数前向声明
void init_mcts_chess_node(const ChessNode& x); // 初始化MCTS棋盘节点
ChessNode uct_search(const ChessNode& x, int player_to_move); // UCT搜索主函数 (center 参数移除, 使用全局 current_search_center)
std::pair<ChessNode, int> tree_policy(const ChessNode& x, int player); // MCTS树策略：选择或扩展节点
ChessNode expand_node(const ChessNode& x, int player); // MCTS扩展节点 (原 expand)
double calculate_ucb(const ChessNode& x, int player); // 计算UCB值 (原 UCB)
std::pair<int, int> calculate_center_of_mass(const ChessNode& x); // 计算棋盘上棋子的质心 (原 cal_centre)
double default_policy_simulation(ChessNode x, int player); // MCTS默认策略：模拟对局 (原 default_policy)
void backpropagate(const ChessNode& start_node, const ChessNode& root_node, double value); // MCTS反向传播 (原 back_up)

// 胜负检查和启发式函数前向声明
int checkWinMCTS(const ChessNode& x, int player_color_mcts); // 检查MCTS棋盘节点中指定颜色是否获胜
bool is_terminal_node(const ChessNode& x); // 检查MCTS棋盘节点是否为终止状态 (棋盘满) (原 is_terminal)
int count_stones_in_region(const ChessNode& x, int r1, int c1, int r2, int c2); // 计算指定区域内的棋子数量 (原 cnt_num)

// Alpha-go/mainWork.cpp 中的启发式函数声明
std::pair<int, std::pair<int, int>> check_immediate_win_or_block(const ChessNode& x, int current_player_color); // 检查一步胜利或阻止对手一步胜利 (合并原 check_four)
std::pair<int, std::pair<int, int>> check_strong_three_threat(const ChessNode& x, int current_player_color); // 检查活三威胁或机会 (合并原 check_three)
bool is_safe_and_empty_mcts(const ChessNode& board, int r, int c); // 检查MCTS棋盘某位置是否安全且为空
bool is_player_stone_mcts(const ChessNode& board, int r, int c, int player_to_check_mcts); // 检查MCTS棋盘某位置是否为指定玩家的棋子
bool is_strong_three_mcts(const ChessNode& board, int player_mcts, int r_new, int c_new); // 检查在(r_new, c_new)落子是否形成活三

// 辅助函数：检查坐标是否在棋盘内
bool inGrid(int r, int c) {
    return r >= 0 && r < SIZE && c >= 0 && c < SIZE;
}

// 辅助函数：检查棋盘是否已满
bool gridFull() {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (Grid[i][j] == -1) { // -1 表示空
                return false;
            }
        }
    }
    return true;
}

// 函数：在全局棋盘上落子
// 成功返回 true，否则 (例如，无效移动) 返回 false
bool placeAt(int player_color, int r, int c) {
    if (inGrid(r, c) && Grid[r][c] == -1) { // 必须在棋盘内且为空位
        Grid[r][c] = player_color;
        return true;
    }
    return false;
}

// 函数：检查在全局棋盘 (r, c) 位置为 player_color 落子后是否获胜
bool checkWin(int player_color, int r, int c) {
    // (r,c) 是刚落下的子. Grid[r][c] 已经是 player_color.
    int dr[] = {1, 0, 1, 1}; // 方向向量: 水平, 垂直, 右下对角线, 右上对角线
    int dc[] = {0, 1, 1, -1}; // 对应的列变化

    for (int i = 0; i < 4; ++i) { // 遍历4个方向
        int count = 1; // 包括刚落的子

        // 检查一个方向 (例如：右, 下, 右下, 右上)
        for (int k = 1; k < 5; ++k) {
            int nr = r + k * dr[i];
            int nc = c + k * dc[i];
            if (inGrid(nr, nc) && Grid[nr][nc] == player_color) {
                count++;
            } else {
                break;
            }
        }

        // 检查相反方向 (例如：左, 上, 左上, 左下)
        for (int k = 1; k < 5; ++k) {
            int nr = r - k * dr[i]; 
            int nc = c - k * dc[i];
            if (inGrid(nr, nc) && Grid[nr][nc] == player_color) {
                count++;
            } else {
                break;
            }
        }
        if (count >= 5) return true; // 达到五子连珠
    }
    return false; // 未获胜
}

// AI 决策逻辑
// current_player_color: 当前轮到哪方下棋 (0: 黑棋, 1: 白棋)
// 返回值为 {行, 列}
std::pair<int, int> decideMove(int current_player_color) {
    // 将当前全局 Grid (std::vector<std::vector<int>>) 转换为 ChessNode
    ChessNode current_board_node(Grid);
    
    // 根据当前棋盘状态更新搜索中心
    current_search_center = calculate_center_of_mass(current_board_node);

    // 调用 MCTS 搜索
    // 玩家颜色映射: gomoku_bot (0:黑, 1:白) 与 MCTS 内部 (0:黑, 1:白) 一致.
    ChessNode best_move_node = uct_search(current_board_node, current_player_color);

    // 找出 current_board_node 和 best_move_node 之间的差异以获取落子位置
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            // 如果原棋盘为空，新棋盘为当前玩家落子，则为AI选择的落子点
            if (current_board_node.g[i][j] == -1 && best_move_node.g[i][j] == current_player_color) {
                return {i, j}; // 返回 {行, 列}
            }
        }
    }

    // 后备策略: 如果MCTS未能找到合适的移动 (理论上在有效棋局中不应发生)
    // 则查找第一个可用的空位 (原始的占位逻辑)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (Grid[i][j] == -1) {
                return {i, j}; 
            }
        }
    }
    return {-1, -1}; // 如果棋盘已满或出现意外情况，返回无效移动
}


// MCTS 及辅助函数的实现 (源自 Alpha-go/mainWork.cpp)
// 已适配 gomoku_bot.cpp 的约定 (Grid: -1 空, 0 黑, 1 白)
// MCTS 内部逻辑的玩家颜色: 0 代表黑棋, 1 代表白棋。

// 初始化MCTS棋盘节点 x 的属性
void init_mcts_chess_node(const ChessNode& x) {
    Property p;
    p.a = 0.0; // 初始化胜利/评估分数为0
    p.b = 0.0; // 初始化访问次数为0
    mcts_mp[x] = std::move(p); // 存入MCTS节点信息表
}

// MCTS 主搜索函数
// x: 当前棋盘状态节点
// player_to_move: 当前轮到哪方下棋 (0: 黑棋, 1: 白棋)
ChessNode uct_search(const ChessNode& x, int player_to_move) {
    auto start_time = std::chrono::steady_clock::now(); // 记录开始时间
    const std::chrono::duration<double> time_limit(0.9); // 设置时间限制为0.9秒

    ChessNode current_node_state = x; // 创建一个可修改的副本
    ChessNode best_next_node = current_node_state; // 如果没有可行的移动，默认返回当前状态

    // 启发式策略：检查 player_to_move 是否能立即获胜
    auto immediate_win_result = check_immediate_win_or_block(current_node_state, player_to_move);
    if (immediate_win_result.first == player_to_move) { // .first 返回的是能获胜的玩家颜色
        best_next_node.g[immediate_win_result.second.first][immediate_win_result.second.second] = player_to_move;
        return best_next_node;
    }
    // 启发式策略：阻止对手立即获胜
    int opponent_player = 1 - player_to_move;
    auto block_opponent_win_result = check_immediate_win_or_block(current_node_state, opponent_player);
     if (block_opponent_win_result.first == opponent_player) {
        best_next_node.g[block_opponent_win_result.second.first][block_opponent_win_result.second.second] = player_to_move; // 我方在对方的致胜点落子
        return best_next_node;
    }

    // 启发式策略：检查 player_to_move 是否能形成活三
    auto strong_three_result = check_strong_three_threat(current_node_state, player_to_move);
    if (strong_three_result.first == player_to_move ) {
         best_next_node.g[strong_three_result.second.first][strong_three_result.second.second] = player_to_move;
         return best_next_node;
    }
    // 启发式策略：阻止对手形成活三
    auto block_opponent_three_result = check_strong_three_threat(current_node_state, opponent_player);
    if (block_opponent_three_result.first == opponent_player) {
        best_next_node.g[block_opponent_three_result.second.first][block_opponent_three_result.second.second] = player_to_move; // 我方在对方的活三点落子
        return best_next_node;
    }

    // 如果当前节点未在MCTS树中，则初始化它
    if (mcts_mp.find(current_node_state) == mcts_mp.end()) {
        init_mcts_chess_node(current_node_state);
    }

    // MCTS主循环：选择、扩展、模拟、反向传播
    for (int cnt = 0; cnt < SELECT_NUM; ++cnt) {
        auto current_time = std::chrono::steady_clock::now();
        if (current_time - start_time > time_limit) {
            break; // 如果超出时间限制，则终止搜索
        }

        // tree_policy 的 player 参数是该节点轮到的玩家
        std::pair<ChessNode, int> selected_leaf_and_player = tree_policy(current_node_state, player_to_move);
        for (int j = 0; j < STA_NUM; j++) { 
            // default_policy_simulation 的 player 参数是在模拟中首先下棋的玩家
            double simulation_score = default_policy_simulation(selected_leaf_and_player.first, selected_leaf_and_player.second);
            // simulation_score: 黑(0)胜为1, 白(1)胜为-1, 平局为0.
            backpropagate(selected_leaf_and_player.first, current_node_state, simulation_score);
        }
    }

    // 如果根节点没有子节点 (可能由于无法扩展或所有子节点都被剪枝，或时间耗尽)
    if (mcts_mp.find(current_node_state) == mcts_mp.end() || mcts_mp[current_node_state].vec.empty()) { // 增加对 current_node_state 是否存在的检查
        // 如果MCTS没有产生任何子节点 (例如时间太短，或者棋盘已满无法扩展)
        // 尝试返回启发式选择的 best_next_node (如果之前有的话)
        // 如果启发式也没有找到，则需要一个后备策略 (例如随机选择一个空位)
        // 当前 best_next_node 可能仍然是 current_node_state
        // 如果 best_next_node 与 current_node_state 相同，说明启发式也没找到，MCTS也没结果
        if (best_next_node == current_node_state) {
             // 后备：随机选择一个可落子点 (如果时间允许，或者作为最后手段)
            std::vector<std::pair<int, int>> empty_cells;
            for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) if(x.g[r][c] == -1) empty_cells.push_back({r,c});
            if(!empty_cells.empty()){
                int move_idx = rand() % empty_cells.size();
                best_next_node.g[empty_cells[move_idx].first][empty_cells[move_idx].second] = player_to_move;
            }
            // 如果棋盘已满，empty_cells会为空，best_next_node 保持 current_node_state，由调用者处理
        }
        return best_next_node;
    }

    // 从子节点中选择最佳着法
    const auto& children = mcts_mp[current_node_state].vec;
    if (!children.empty()) { // 应该总是不为空，除非MCTS提前终止且未产生子节点
        best_next_node = children.front(); // 初始化为第一个子节点
        // UCB值的计算是基于 player_to_move (在根节点x做决策的玩家) 的视角
        double max_ucb = calculate_ucb(best_next_node, player_to_move);

        for (const auto& child_node : children) {
            double ucb_val = calculate_ucb(child_node, player_to_move);
            if (ucb_val > max_ucb) {
                max_ucb = ucb_val;
                best_next_node = child_node;
            }
        }
    } else { // 如果时间耗尽导致 children 为空 (理论上不太可能，因为上面有检查)
        if (best_next_node == current_node_state) { // 再次检查，确保有后备
            std::vector<std::pair<int, int>> empty_cells;
            for(int r=0; r<SIZE; ++r) for(int c=0; c<SIZE; ++c) if(x.g[r][c] == -1) empty_cells.push_back({r,c});
            if(!empty_cells.empty()){
                int move_idx = rand() % empty_cells.size();
                best_next_node.g[empty_cells[move_idx].first][empty_cells[move_idx].second] = player_to_move;
            }
        }
    }
    
    // 每次uct_search调用后清理MCTS树，为裁判的下一轮调用做准备
    // (原game_window循环中的清理逻辑)
    mcts_mp.clear(); 
    mcts_fa.clear();

    return best_next_node;
}

// MCTS树策略：选择或扩展节点
// x: 当前节点状态
// player: 在当前节点 x 轮到哪方下棋
std::pair<ChessNode, int> tree_policy(const ChessNode& x, int player) {
    ChessNode current_node = x;
    int current_player_at_node = player;

    // 循环直到遇到终止节点 (获胜或棋盘满) 或可扩展的节点
    while (!checkWinMCTS(current_node, 0) && !checkWinMCTS(current_node,1) && !is_terminal_node(current_node)) {
        // current_search_center 是全局变量, 在 uct_search 调用前已更新
        int r1 = std::max(0, current_search_center.first - SEARCH_RANGE);
        int r2 = std::min(SIZE -1 , current_search_center.first + SEARCH_RANGE); 
        int c1 = std::max(0, current_search_center.second - SEARCH_RANGE);
        int c2 = std::min(SIZE -1, current_search_center.second + SEARCH_RANGE); 
        
        // 确保节点在 MCTS 节点信息表中
        if (mcts_mp.find(current_node) == mcts_mp.end()) {
            init_mcts_chess_node(current_node);
        }

        // 扩展条件判断：是否可以在搜索范围内找到一个空位进行扩展，且该空位未成为当前节点的子节点
        bool can_expand = false;
        // 如果子节点数量小于搜索范围内的最大可能数量，则可能可以扩展
        if (mcts_mp[current_node].vec.size() < (size_t)((r2 - r1 + 1) * (c2 - c1 + 1))) {
            for (int i = r1; i <= r2; ++i) {
                for (int j = c1; j <= c2; ++j) {
                    if (current_node.g[i][j] == -1) { // 如果是空位
                        ChessNode temp_child = current_node;
                        temp_child.g[i][j] = current_player_at_node; // 假设在此处落子
                        bool child_exists = false; // 检查这个假设的子节点是否已存在
                        for(const auto& existing_child : mcts_mp[current_node].vec){
                            if(existing_child == temp_child){
                                child_exists = true;
                                break;
                            }
                        }
                        if(!child_exists){ // 如果不存在，则可以扩展
                            can_expand = true;
                            break;
                        }
                    }
                }
                if(can_expand) break;
            }
        }

        if (can_expand) { // 如果可以扩展
            return MP(expand_node(current_node, current_player_at_node), current_player_at_node);
        } else { // 如果不能扩展 (搜索范围内没有新的可落子点，或子节点已满)
            if (mcts_mp[current_node].vec.empty()) { // 没有子节点且无法扩展 (例如棋盘满或搜索范围限制)
                return MP(current_node, current_player_at_node); // 从当前节点开始模拟
            }

            // 选择UCB值最大的子节点
            const auto& children = mcts_mp[current_node].vec;
            ChessNode best_child_node = children.front();
            // UCB的计算是基于父节点 (current_node) 的玩家 (current_player_at_node) 的视角
            double max_ucb = calculate_ucb(best_child_node, current_player_at_node); 

            for (const auto& child : children) {
                double ucb_val = calculate_ucb(child, current_player_at_node);
                if (ucb_val > max_ucb) {
                    max_ucb = ucb_val;
                    best_child_node = child;
                }
            }
            mcts_fa[best_child_node] = current_node; //记录父节点
            current_node = best_child_node; //深入到最佳子节点
        }
        current_player_at_node = 1 - current_player_at_node; // 切换到下一层的玩家
    }
    return MP(current_node, current_player_at_node); // 返回终止节点和轮到下棋的玩家
}

// MCTS扩展节点
// x: 父节点状态
// player: 在父节点 x 轮到哪方下棋 (即将在新扩展的子节点上落子的玩家)
ChessNode expand_node(const ChessNode& x, int player) {
    ChessNode y = x; // 可修改的副本

    // 搜索范围，同 tree_policy
    int r1 = std::max(0, current_search_center.first - SEARCH_RANGE);
    int r2 = std::min(SIZE - 1, current_search_center.first + SEARCH_RANGE);
    int c1 = std::max(0, current_search_center.second - SEARCH_RANGE);
    int c2 = std::min(SIZE - 1, current_search_center.second + SEARCH_RANGE);

    std::vector<std::pair<int,int>> possible_moves; // 存储搜索范围内的可落子点
    for(int i=r1; i<=r2; ++i){
        for(int j=c1; j<=c2; ++j){
            if(y.g[i][j] == -1){ // 如果是空位
                ChessNode temp_child = y;
                temp_child.g[i][j] = player; // 假设落子
                bool child_exists = false;
                if (mcts_mp.count(x)) { 
                    for(const auto& existing_child : mcts_mp[x].vec){
                        if(existing_child == temp_child){
                            child_exists = true;
                            break;
                        }
                    }
                }
                if(!child_exists){ // 如果不是现有子节点，则为可扩展的移动
                    possible_moves.push_back({i,j});
                }
            }
        }
    }

    if(possible_moves.empty()){ // 如果在限定搜索范围内没有可扩展的移动
        // 尝试在整个棋盘上寻找可扩展的移动 (后备策略)
        for(int i=0; i<SIZE; ++i) {
            for(int j=0; j<SIZE; ++j) {
                 if(y.g[i][j] == -1){ // 空位
                    ChessNode temp_child = y;
                    temp_child.g[i][j] = player;
                    bool child_exists = false;
                    if (mcts_mp.count(x)) {
                        for(const auto& existing_child : mcts_mp[x].vec){
                            if(existing_child == temp_child){
                                child_exists = true;
                                break;
                            }
                        }
                    }
                    if(!child_exists){
                        possible_moves.push_back({i,j});
                    }
                 }
            }
        }
        if(possible_moves.empty()) return x; // 如果整个棋盘都没有可扩展的移动 (例如棋盘已满)，返回原节点
    }

    // 从可能的移动中随机选择一个进行扩展
    int move_idx = rand() % possible_moves.size();
    std::pair<int,int> chosen_move = possible_moves[move_idx];
    y.g[chosen_move.first][chosen_move.second] = player; // 在副本上执行落子

    // 初始化新扩展的节点 y
    if (mcts_mp.find(y) == mcts_mp.end()) {
        init_mcts_chess_node(y);
    }
    // 将新节点 y 添加到父节点 x 的子节点列表，并记录父子关系
    if (mcts_mp.find(x) == mcts_mp.end()) { // 父节点 x 应该已被 tree_policy 初始化
        init_mcts_chess_node(x);
    }
    mcts_mp[x].vec.push_back(y);
    mcts_fa[y] = x;
    return y; // 返回新扩展的子节点状态
}

// 计算子节点 child_node 的UCB1值
// parent_player: 父节点的玩家 (即做决策选择此 child_node 的玩家)
double calculate_ucb(const ChessNode& child_node, int parent_player) {
    // 如果子节点未被访问或访问次数为0，给予高优先级 (INF) 以鼓励探索
    if (mcts_mp.find(child_node) == mcts_mp.end() || mcts_mp[child_node].b == 0) {
        return INF; 
    }

    double wins = mcts_mp[child_node].a;   // 子节点的累计得分 (黑胜为正，白胜为负)
    double visits = mcts_mp[child_node].b; // 子节点的访问次数

    // Exploitation Term (利用项): 根据父节点玩家调整得分视角
    // parent_player 为黑(0), 希望最大化 wins/visits
    // parent_player 为白(1), 希望最大化 -wins/visits (即最小化 wins/visits)
    double exploitation_term;
    if (parent_player == 0) { // 黑棋视角
        exploitation_term = wins / visits;
    } else { // 白棋视角
        exploitation_term = -wins / visits;
    }

    // Exploration Term (探索项): C * sqrt(log(N_parent) / N_child)
    double exploration_term = 0.0; 
    ChessNode parent_of_child;
    bool parent_exists = false;
    if(mcts_fa.count(child_node)) { // 获取父节点
        parent_of_child = mcts_fa[child_node];
        parent_exists = true;
    }

    // 使用父节点的访问次数计算探索项
    if (parent_exists && mcts_mp.count(parent_of_child) && mcts_mp[parent_of_child].b > 0) {
         // C (常数) 通常取 sqrt(2)
         exploration_term = sqrt(2.0 * log(mcts_mp[parent_of_child].b) / visits); 
    } else if (visits > 0) { // 如果父节点信息不完整或父节点访问为0 (理论上不应发生于已访问子节点)
         // 后备探索项，使用总选择次数作为父访问次数的近似
         exploration_term = sqrt(2.0 * log(static_cast<double>(SELECT_NUM)) / visits); 
    }
    // 如果 visits 为0, exploration_term 也应为0 (因为函数开头会返回 INF)

    return exploitation_term + exploration_term;
}


// 计算棋盘 x 上棋子的质心
std::pair<int, int> calculate_center_of_mass(const ChessNode& x) {
    int stone_count = 0;    // 棋子总数
    double sum_r = 0, sum_c = 0; // 行、列坐标总和 (用double避免精度损失)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (x.g[i][j] != -1) { // 如果不是空位
                stone_count++;
                sum_r += i;
                sum_c += j;
            }
        }
    }
    if (stone_count == 0) { // 如果棋盘为空
        return MP(SIZE / 2, SIZE / 2); // 返回棋盘中心
    }
    // 返回四舍五入后的平均坐标作为质心
    return MP(static_cast<int>(round(sum_r / stone_count)), static_cast<int>(round(sum_c / stone_count)));
}

// MCTS默认策略：快速模拟对局直到结束
// current_board_sim: 开始模拟的棋盘状态 (副本)
// player_to_move_sim: 在模拟中首先下棋的玩家
// 返回值: 黑(0)胜为1.0, 白(1)胜为-1.0, 平局为0.0
double default_policy_simulation(ChessNode current_board_sim, int player_to_move_sim) {
    ChessNode board_copy = current_board_sim; // 在副本上操作
    int current_player_in_sim = player_to_move_sim;

    const int MAX_SIMULATION_DEPTH = SIZE * SIZE + 5; // 模拟的最大深度 (避免无限循环), 略大于棋盘格子数
    int depth = 0;

    while (depth < MAX_SIMULATION_DEPTH) {
        if (checkWinMCTS(board_copy, 0)) return 1.0;  // 黑(0)胜
        if (checkWinMCTS(board_copy, 1)) return -1.0; // 白(1)胜
        if (is_terminal_node(board_copy)) return 0.0; // 平局 (棋盘满)

        // 随机选择一个空位落子
        std::vector<std::pair<int, int>> empty_cells;
        // 启发式：优先在当前质心附近模拟，可适当扩大范围
        std::pair<int,int> sim_center = calculate_center_of_mass(board_copy);
        int r1 = std::max(0, sim_center.first - SEARCH_RANGE - 1); 
        int r2 = std::min(SIZE - 1, sim_center.first + SEARCH_RANGE + 1);
        int c1 = std::max(0, sim_center.second - SEARCH_RANGE - 1);
        int c2 = std::min(SIZE - 1, sim_center.second + SEARCH_RANGE + 1);

        for (int r = r1; r <= r2; ++r) { // 优先在质心附近找空位
            for (int c = c1; c <= c2; ++c) {
                if (board_copy.g[r][c] == -1) {
                    empty_cells.push_back(MP(r, c));
                }
            }
        }
        if(empty_cells.empty()){ // 如果质心附近没有空位，则在整个棋盘上找
            for (int r = 0; r < SIZE; ++r) {
                for (int c = 0; c < SIZE; ++c) {
                    if (board_copy.g[r][c] == -1) {
                        empty_cells.push_back(MP(r, c));
                    }
                }
            }
        }

        if (empty_cells.empty()) return 0.0; // 没有可落子点 (理论上会被 is_terminal_node 捕获)

        int move_idx = rand() % empty_cells.size(); // 随机选择一个空位
        std::pair<int, int> chosen_move = empty_cells[move_idx];
        board_copy.g[chosen_move.first][chosen_move.second] = current_player_in_sim; // 落子

        current_player_in_sim = 1 - current_player_in_sim; // 切换玩家
        depth++;
    }
    return 0.0; // 达到最大模拟深度，视为平局
}

// MCTS反向传播：从叶节点 leaf_node 向上更新到根节点 root_node_of_search
// value: 模拟结果 (黑胜1, 白胜-1, 平局0)
void backpropagate(const ChessNode& leaf_node, const ChessNode& root_node_of_search, double value) {
    ChessNode current = leaf_node;
    while (true) {
        if (mcts_mp.find(current) == mcts_mp.end()) { // 节点不存在 (理论上不应发生，但增加保护)
            break; 
        }
        mcts_mp[current].a += value; // 更新累计得分
        mcts_mp[current].b++;        // 更新访问次数

        // 如果到达搜索树的根节点或找不到父节点，则停止传播
        if (current == root_node_of_search || mcts_fa.find(current) == mcts_fa.end()) {
            break;
        }
        if (mcts_fa[current] == current) { // 避免因父节点错误指向自身导致死循环 (理论上不应发生)
            break;
        }
        current = mcts_fa[current]; // 移动到父节点
    }
}

// 检查MCTS棋盘节点 x 是否为终止状态 (棋盘是否已满)
bool is_terminal_node(const ChessNode& x) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (x.g[i][j] == -1) { // 只要有空位，就不是终止状态
                return false;
            }
        }
    }
    return true; // 棋盘已满
}

// 计算MCTS棋盘节点 x 在指定区域 [r1, c1] 到 [r2, c2] 内的棋子数量 (非空位置数量)
int count_stones_in_region(const ChessNode& x, int r1, int c1, int r2, int c2) {
    int sum = 0;
    // 确保坐标在有效范围内
    r1 = std::max(0, r1);
    r2 = std::min(SIZE - 1, r2);
    c1 = std::max(0, c1);
    c2 = std::min(SIZE - 1, c2);

    for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
            if (x.g[i][j] != -1) sum++; // 如果不是空位，则计数
        }
    }
    return sum;
}

// 检查MCTS棋盘节点 x 中，指定颜色的玩家 player_color_mcts (0黑, 1白) 是否获胜
// 返回1表示获胜，0表示未获胜
int checkWinMCTS(const ChessNode& x, int player_color_mcts) {
    // 水平方向检查
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j <= SIZE - 5; j++) { // 优化：内层循环只需到 SIZE - 5 即可           
            int count = 0;
            for(int k=0; k<5; ++k){
                if (x.g[i][j+k] == player_color_mcts) count++;
                else break;
            }
            if (count == 5) return 1;
        }
    }
    // 垂直方向检查
    for (int j = 0; j < SIZE; j++) {
        for (int i = 0; i <= SIZE - 5; i++) { // 优化：内层循环只需到 SIZE - 5 即可      
            int count = 0;
            for(int k=0; k<5; ++k){
                if (x.g[i+k][j] == player_color_mcts) count++;
                else break;
            }
            if (count == 5) return 1;
        }
    }
    // 主对角线方向检查 (左上到右下)
    for (int i = 0; i <= SIZE - 5; i++) {
        for (int j = 0; j <= SIZE - 5; j++) {
            int count = 0;
            for (int k = 0; k < 5; k++) {
                if (x.g[i + k][j + k] == player_color_mcts) count++;
                else break;
            }
            if (count == 5) return 1;
        }
    }
    // 副对角线方向检查 (右上到左下)
    for (int i = 0; i <= SIZE - 5; i++) {
        for (int j = 4; j < SIZE; j++) { 
            int count = 0;
            for (int k = 0; k < 5; k++) {
                if (x.g[i + k][j - k] == player_color_mcts) count++;
                else break;
            }
            if (count == 5) return 1;
        }
    }
    return 0; // 未获胜
}


// 启发式函数 (源自 Alpha-go/mainWork.cpp，已适配)
// 这些函数中的玩家颜色:
// - 输入参数 current_player_color (来自 decideMove) 是 0代表黑, 1代表白.
// - ChessNode g[][] 中的棋子表示: -1空, 0黑, 1白.
// - 启发式函数内部的 player 变量通常与此一致 (0黑, 1白).
// - 返回值 pair.first: -1表示无威胁/机会; 0表示黑方有; 1表示白方有.

// 检查MCTS棋盘 board 的 (r, c) 位置是否安全(在棋盘内)且为空(-1)
bool is_safe_and_empty_mcts(const ChessNode& board, int r, int c) {
    if (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
        return board.g[r][c] == -1; // -1 代表空
    }
    return false;
}

// 检查MCTS棋盘 board 的 (r, c) 位置是否为指定玩家 player_to_check_mcts (0黑, 1白) 的棋子
bool is_player_stone_mcts(const ChessNode& board, int r, int c, int player_to_check_mcts) {
    if (r >= 0 && r < SIZE && c >= 0 && c < SIZE) {
        return board.g[r][c] == player_to_check_mcts;
    }
    return false;
}


// 检查 player_for_threat (0黑, 1白) 是否能在棋盘 x 上一步获胜 (形成五子连珠)
// 返回: {能获胜的玩家颜色, {行, 列}} 如果找到获胜点.
//       {-1, {-1,-1}} 否则. // 将无效位置改为-1,-1
// 此函数替代原 check_four.
std::pair<int, std::pair<int, int>> check_immediate_win_or_block(const ChessNode& x, int player_for_threat) {
    ChessNode temp_board = x; // 在临时棋盘上操作
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (temp_board.g[i][j] == -1) { // 如果是空位
                temp_board.g[i][j] = player_for_threat; // 假设落子
                if (checkWinMCTS(temp_board, player_for_threat)) { // 检查是否获胜
                    // temp_board.g[i][j] = -1; // 恢复棋盘 (对于局部副本非必需, 因为每次循环都从x复制)
                    return MP(player_for_threat, MP(i, j)); // 返回能获胜的玩家和位置
                }
                temp_board.g[i][j] = -1; // 恢复棋盘状态，继续检查下一个空位
            }
        }
    }
    return MP(-1, MP(-1, -1)); // 未找到一步制胜点, 返回无效坐标
}


// 检查在 board_before_move 的 (r_new, c_new) 位置为 player_mcts (0黑, 1白) 落子后是否形成活三
// 假设 (r_new, c_new) 当前为空，player_mcts 的棋子被假设放置在那里。
bool is_strong_three_mcts(const ChessNode& board_before_move, int player_mcts, int r_new, int c_new) {
    // ChessNode board_with_move = board_before_move; // 不需要创建完整副本，直接在判断时考虑新子
    // board_with_move.g[r_new][c_new] = player_mcts; // 假设落子

    int dr[] = {0, 1, 1, 1}; // 方向向量: 水平, 垂直, 右下对角线, 右上对角线
    int dc[] = {1, 0, 1, -1};

    for (int i = 0; i < 4; ++i) { // 遍历4个方向
        // 检查三种活三模式 (X代表新落子在(r_new,c_new), O代表己方已存在的子, .代表空位)
        // 模式 1: . X O O .  (X是新子, 即在 (r_new, c_new) )
        // 空位在 (r_new - dr[i], c_new - dc[i]) [基于board_before_move]
        // O 在 (r_new + dr[i], c_new + dc[i]) [基于board_before_move]
        // O 在 (r_new + 2*dr[i], c_new + 2*dc[i]) [基于board_before_move]
        // 空位在 (r_new + 3*dr[i], c_new + 3*dc[i]) [基于board_before_move]
        if (is_safe_and_empty_mcts(board_before_move, r_new - dr[i], c_new - dc[i]) &&
            is_player_stone_mcts(board_before_move, r_new + dr[i], c_new + dc[i], player_mcts) &&
            is_player_stone_mcts(board_before_move, r_new + 2 * dr[i], c_new + 2 * dc[i], player_mcts) &&
            is_safe_and_empty_mcts(board_before_move, r_new + 3 * dr[i], c_new + 3 * dc[i])) {
            return true;
        }
        // 模式 2: . O X O . (X是新子)
        // 空位在 (r_new - 2*dr[i], c_new - 2*dc[i])
        // O 在 (r_new - dr[i], c_new - dc[i])
        // O 在 (r_new + dr[i], c_new + dc[i])
        // 空位在 (r_new + 2*dr[i], c_new + 2*dc[i])
        if (is_safe_and_empty_mcts(board_before_move, r_new - 2*dr[i], c_new - 2*dc[i]) &&
            is_player_stone_mcts(board_before_move, r_new - dr[i], c_new - dc[i], player_mcts) &&
            is_player_stone_mcts(board_before_move, r_new + dr[i], c_new + dc[i], player_mcts) &&
            is_safe_and_empty_mcts(board_before_move, r_new + 2*dr[i], c_new + 2*dc[i])) {
            return true;
        }
        // 模式 3: . O O X . (X是新子)
        // 空位在 (r_new + dr[i], c_new + dc[i])
        // O 在 (r_new - dr[i], c_new - dc[i])
        // O 在 (r_new - 2*dr[i], c_new - 2*dc[i])
        // 空位在 (r_new - 3*dr[i], c_new - 3*dc[i])
         if (is_safe_and_empty_mcts(board_before_move, r_new + dr[i], c_new + dc[i]) && // 这是点位 (r_new + dr[i], c_new + dc[i]) 应该为空
            is_player_stone_mcts(board_before_move, r_new - dr[i], c_new - dc[i], player_mcts) &&
            is_player_stone_mcts(board_before_move, r_new - 2 * dr[i], c_new - 2 * dc[i], player_mcts) &&
            is_safe_and_empty_mcts(board_before_move, r_new - 3 * dr[i], c_new - 3 * dc[i])) { // 这是点位 (r_new - 3*dr[i], c_new - 3*dc[i]) 应该为空
            return true;
        }
    }
    return false; // 未找到活三
}

// 检查 player_for_threat (0黑, 1白) 是否能在棋盘 x 上形成活三
// 返回: {能形成活三的玩家颜色, {行, 列}} 如果找到.
//       {-1, {-1,-1}} 否则. // 将无效位置改为-1,-1
// 此函数替代原 check_three.
std::pair<int, std::pair<int, int>> check_strong_three_threat(const ChessNode& x, int player_for_threat) {
    ChessNode temp_board = x; // 原始棋盘状态
    std::vector<std::pair<int,int>> potential_three_moves; // 存储所有能形成活三的落子点

    for (int r = 0; r < SIZE; ++r) {
        for (int c = 0; c < SIZE; ++c) {
            if (temp_board.g[r][c] == -1) { // 如果是空位
                if (is_strong_three_mcts(temp_board, player_for_threat, r, c)) { // 检查在此处落子是否形成活三
                    potential_three_moves.push_back(MP(r,c));
                }
            }
        }
    }

    if (!potential_three_moves.empty()) { // 如果找到一个或多个活三点
        // 如果有多个选择，可以选择一个 (例如随机，或最接近中心的)
        // 为简单起见，选择第一个找到的，或像原 check_three 一样选择最接近中心的
        if (potential_three_moves.size() == 1) {
            return MP(player_for_threat, potential_three_moves.front());
        }
        // 选择最接近质心的点 (同原 check_three 的 choose_best_move 逻辑)
        auto center_pos = calculate_center_of_mass(x); // 计算当前棋盘质心
        int min_dist_sq = SIZE*SIZE*2 +1; // 一个足够大的初始最小距离的平方 (避免sqrt)
        std::pair<int, int> best_move = potential_three_moves.front();
        for (const auto& move : potential_three_moves) {
            int dr_center = move.first - center_pos.first;
            int dc_center = move.second - center_pos.second;
            int dist_sq = dr_center * dr_center + dc_center * dc_center; // 曼哈顿距离的平方
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_move = move;
            }
        }
        return MP(player_for_threat, best_move); // 返回能形成活三的玩家和最佳位置
    }

    return MP(-1, MP(-1, -1)); // 未找到活三威胁点, 返回无效坐标
}


// 主函数：处理与裁判程序的交互
int main() {
    srand(static_cast<unsigned int>(time(NULL))); // 初始化随机数种子

    std::ios_base::sync_with_stdio(false); // 加速C++标准流
    std::cin.tie(NULL); // 解除 cin 和 cout 的绑定

    std::string line;
    // 从标准输入读取一行JSON字符串
    if (!std::getline(std::cin, line)) {
        // std::cerr << "错误：无法从标准输入读取数据。" << std::endl; // 错误：无法从标准输入读取数据。
        return 1; // 如果无法读取输入则退出
    }

    Json::Value input_json; // 使用 Json::Value
    Json::Reader reader;    // 使用 Json::Reader 进行解析
    
    // 检查输入行是否为空，然后尝试解析
    if (line.empty()) {
        // std::cerr << "错误：接收到空的输入行。" << std::endl;
        return 1; // 空输入行，作错误处理
    }
    if (!reader.parse(line, input_json)) {
        // std::cerr << "JSON 解析错误。 输入: " << line << std::endl; // JSON 解析错误
        // std::cerr << "Reader errors: " << reader.getFormattedErrorMessages() << std::endl; // 输出更详细的jsoncpp解析错误
        return 1; // JSON解析失败则退出
    }

    // 清空全局棋盘状态
    for(int r_idx = 0; r_idx < SIZE; ++r_idx) {
        std::fill(Grid[r_idx].begin(), Grid[r_idx].end(), -1); // -1 代表空
    }

    bool is_bot_black = false; // Bot是否执黑，默认为否 (执白)
    int turn_id = 0; // 代表接收到的 requests 数组的大小 (即对手已走的步数，或我方作为黑棋时的标记)

    // 从 input_json["requests"] 判断Bot颜色和回合数
    if (input_json.isMember("requests") && input_json["requests"].isArray()) {
        const Json::Value& requests_array = input_json["requests"];
        turn_id = requests_array.size(); // 获取请求数组的大小，用于判断回合

        if (turn_id > 0) {
            const Json::Value& first_request = requests_array[0u]; // 使用 0u 避免警告, JsonCpp索引
            if (first_request.isObject() && first_request.isMember("x") && 
                first_request["x"].isInt() && first_request["x"].asInt() == -1) {
                is_bot_black = true; // 如果是，则Bot执黑
            }
        }
    } else {
        // std::cerr << "错误：输入JSON中 \'requests\' 字段缺失或格式不正确。" << std::endl; 
        return 1; // 异常退出
    }

    int bot_color = is_bot_black ? 0 : 1; // Bot的棋子颜色 (0: 黑棋, 1: 白棋)
    int opponent_color = is_bot_black ? 1 : 0; // 对手的棋子颜色

    // 特殊情况：如果Bot执黑且是第一步 (turn_id 为1，因为requests[0]是{-1,...}标记)
    if (is_bot_black && turn_id == 1) {
        Json::Value output_json_first_move; 
        output_json_first_move["response"]["x"] = SIZE / 2; 
        output_json_first_move["response"]["y"] = SIZE / 2;
        
        Json::FastWriter writer; 
        std::cout << writer.write(output_json_first_move); 
        return 0; 
    }

    // 根据历史记录重建棋盘状态
    if (input_json.isMember("requests") && input_json["requests"].isArray()) {
        const Json::Value& requests_log = input_json["requests"]; 
        const Json::Value* responses_log_ptr = nullptr; 
        if (input_json.isMember("responses") && input_json["responses"].isArray()) {
            responses_log_ptr = &input_json["responses"];
        }

        // 遍历 turn_id 次，turn_id 是 requests 数组的大小
        // 这个循环的目的是根据裁判提供的历史记录，在Bot的内部棋盘上重现局面
        for (int i = 0; i < turn_id; ++i) {
            // 处理对手的落子 (来自 requests[i])
            if (!is_bot_black || i > 0) { 
                // 确保索引 i 在 requests_log 的有效范围内
                if (i < (int)requests_log.size() && requests_log[i].isObject() &&
                    requests_log[i].isMember("x") && requests_log[i]["x"].isInt() &&
                    requests_log[i].isMember("y") && requests_log[i]["y"].isInt()) {
                    
                    int r = requests_log[i]["x"].asInt();
                    int c = requests_log[i]["y"].asInt();
                    placeAt(opponent_color, r, c); // 在棋盘上记录对手的落子
                } else {
                    // std::cerr << "警告：requests[" << i << "] 格式错误或丢失。" << std::endl; 
                }
            }

            if (i == turn_id - 1) {
                break;
            }

            // 确保索引 i 在 responses_log_ptr (如果非空) 的有效范围内
            if (responses_log_ptr && i < (int)responses_log_ptr->size()) {
                 const Json::Value& resp_move = (*responses_log_ptr)[i]; // JsonCpp索引
                 if (resp_move.isObject() &&
                     resp_move.isMember("x") && resp_move["x"].isInt() &&
                     resp_move.isMember("y") && resp_move["y"].isInt()) {
                    
                    int r_resp = resp_move["x"].asInt();
                    int c_resp = resp_move["y"].asInt();
                    placeAt(bot_color, r_resp, c_resp); // 在棋盘上记录Bot之前的落子
                 } else {
                    // std::cerr << "警告：responses[" << i << "] 格式错误或丢失。" << std::endl; 
                 }
            } else if (i < turn_id -1) { 
                // std::cerr << "警告：responses[" << i << "] 缺失。" << std::endl; 
            }
        }
    }

    // AI进行决策
    Json::Value output_json_move; // 使用 Json::Value
    std::pair<int, int> my_move;

    my_move = decideMove(bot_color); 

    output_json_move["response"]["x"] = my_move.first;  // AI决策的行
    output_json_move["response"]["y"] = my_move.second; // AI决策的列

    Json::FastWriter writer; // 使用 Json::FastWriter
    std::cout << writer.write(output_json_move); // 输出决策结果

    return 0; // 程序正常结束
}
