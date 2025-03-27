#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <set>
#include <utility>
#include <functional>
#include <numeric>
#include <iostream>
#include <cassert>
#include <ctime>
#include <random>

#define MAX_DISK_NUM (10 + 1)
#define MAX_DISK_SIZE (16384 + 1)
#define MAX_REQUEST_NUM (30000000 + 1)
#define MAX_OBJECT_NUM (100000 + 1)
#define REP_NUM (3)
#define FRE_PER_SLICING (1800)
#define EXTRA_TIME (105)
#define MAX_OBJECT_SIZE (5 + 1)
#define MAX_TAG_NUM (16 + 1)
#define MAX_STAGE (55)
#define MAX_TOKEN (1000 + 2)
#define MAX_PIECE_QUEUE (105 + 1)
#define INF_TOKEN (10000000)

const double JUMP_VISCOSITY = 0.9;
const int CUR_REQUEST_DIVIDE = 200;
const int LEN_TIME_DIVIDE = 40;
const int PRE_DISTRIBUTION_TIME = 35;
const int READ_CNT_STATES = 8; //读入的状态，根据上一次连续read的个数确定
int DISK_MIN_PASS = 9; //如果超过这个值放弃read pass过去
int DISK_MIN_PASS_DP = 13;
const int MIN_TOKEN_STOP_DP = 130;
const int NUM_PIECE_QUEUE = 2;
const double TAG_DENSITY_DIVIDE = 2;
const double UNIT_REQUEST_DIVIDE = 17;
const int MIN_ROUND_TIME = 3;
const int MIN_TEST_DENSITY_LEN = 500;
const int TEST_READ_TIME = 10;
const double DIVIDE_TAG_IN_DISK_VERSION1 = 0.1;

//这三个量需要调整   需要退火
const int WRITE_TEST_DENSITY_LEN = 50;
const int WRITE_TAG_DENSITY_DIVIDE = 30;
const int MIN_TEST_TAG_DENSITY_LEN = 50;

const int USE_NEW_DISTRIBUTION = 1;
//不要调
const int USE_DP = 1;
const int DP_VERSION1 = 1;
const int DP_VERSION2 = 2;
const int MIN_TAG_NUM_IN_DISK = 6;
//int READ_ROUND_TIME = 40; //一轮读取的时间
const int READ_ROUND_TIME = 3;
int TEST_DENSITY_LEN = 1200;

struct _Object {
    //(磁盘编号，磁盘内位置)
    std::pair <int, int> unit_pos[REP_NUM + 1][MAX_OBJECT_SIZE];
    int size;
    int tag;
};

struct Sub_object {
    int tag;
    int size;
    Sub_object() {}
    Sub_object(int _tag, int _size) : tag(_tag), size(_size) {}
};

struct _Request {
    int object_id;
    int request_time;
    int request_id;
};

int tag_size_in_disk[MAX_TAG_NUM][MAX_DISK_NUM];

_Request requests[MAX_REQUEST_NUM];
std::queue <_Request> request_queue_in_time_order[MAX_PIECE_QUEUE];
std::vector<int> time_out_of_queue(MAX_PIECE_QUEUE);
std::vector<int> request_queue_id(MAX_REQUEST_NUM);

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意，objects加了复数
_Object objects[MAX_OBJECT_NUM];

int T, M, N, V, G, all_stage, now_stage, cur_request;

inline int get_pre_kth(int x, int k) 
{
    x = (x - k + V) % V;
    if (x == 0) x = V;
    return x;
}

inline int get_nxt_kth(int x, int k) {
    x = (x + k) % V;
    if (x == 0) x = V;
    return x;
}

struct Segment_tree_max {
    
    bool preference_left;
    struct Node {
        int max, pos;
        bool preference_left;
        Node() {}
        Node(int _max, int _pos) : max(_max), pos(_pos) {}
        friend Node operator + (const Node& x, const int& num) {
            return Node(x.max + num, x.pos);
        }

        friend bool operator < (const Node& x, const Node& y) {
            if (x.preference_left)
                return x.max < y.max;
            return x.max <= y.max;
        }
    };

    
    int window_len = WRITE_TEST_DENSITY_LEN;

    Segment_tree_max() {}
    Segment_tree_max(int n) {
        seg = std::vector <Node> (n << 2);
        add_tag = std::vector <int> (n << 2, 0);
    }

    std::vector <Node> seg;
    std::vector <int> add_tag;

    inline void init(int n) {
        seg = std::vector <Node> (n << 2);
        add_tag = std::vector <int> (n << 2, 0);
    }

    inline void push_up(int o) 
    {
        // if (seg[o << 1].max >= seg[o << 1 | 1].max)
        //     seg[o] = seg[o << 1];
        // else seg[o] = seg[o << 1 | 1];
        seg[o] = std::max(seg[o << 1], seg[o << 1 | 1]);
    }

    void reset(int o = 1, int l = 1, int r = V)
    {
        add_tag[o] = seg[o].max = 0;
        if (l == r) return ;
        int mid = l + r >> 1;
        reset(o << 1, l, mid);
        reset(o << 1 | 1, mid + 1, r);
        push_up(o);
    }

    void build(int o = 1, int l = 1, int r = V) {
        add_tag[o] = 0;
        seg[o].preference_left = preference_left;
        if (l == r) {
            seg[o].max = 0;
            seg[o].pos = l;
            return ;
        }

        int mid = l + r >> 1;
        build(o << 1, l, mid);
        build(o << 1 | 1, mid + 1, r);
        push_up(o);
    }

    inline void apply(int o, int now_ad) 
    {
        add_tag[o] += now_ad;
        seg[o] = seg[o] + now_ad;
    }

    inline void push_down(int o) 
    {
        if (!add_tag[o]) return ;
        apply(o << 1, add_tag[o]);
        apply(o << 1 | 1, add_tag[o]);
        add_tag[o] = 0;
    }
    
    //返回差值，方便后面维护density
    int modify(int o, int l, int r, int p, int v) 
    {
        if (l == r) {
            int pre = seg[o].max;
            seg[o] = Node(v, l);
            return v - pre;
        }
        
        push_down(o);
        int mid = l + r >> 1, res = 0;
        if (p <= mid)
            res = modify(o << 1, l, mid, p, v);
        else res = modify(o << 1 | 1, mid + 1, r, p, v);

        push_up(o);
        return res;
    }

    void add(int o, int l, int r, int x, int y, int ad) 
    {
        // std::cerr << o << " " << l << " " << r << std::endl;
        if (x <= l && y >= r) {
            return apply(o, ad);
        }

        int mid = l + r >> 1;
        push_down(o);
        if (x <= mid)
            add(o << 1, l, mid, x, y, ad);
        if (y > mid)
            add(o << 1 | 1, mid + 1, r, x, y, ad);
        push_up(o);
    }

    Node query_max(int o, int l, int r, int x, int y) 
    {
        if (x <= l && y >= r) return seg[o];
        push_down(o);
        int mid = l + r >> 1;
        Node res = Node(-1, -1);

        if (x <= mid)
            res = query_max(o << 1, l, mid, x, y);
        if (y > mid)
            res = std::max(res, query_max(o << 1 | 1, mid + 1, r, x, y));
        return res;    
    }

    int find_max_point() {
        return query_max(1, 1, V, 1, V).pos;
    }

    int find_max_value() {
        return query_max(1, 1, V, 1, V).max;
    }

    int find_next(int o, int l, int r, int x, int y, int lim) 
    {
        if (seg[o].max < lim) return -1;
        if (l == r) return l;

        int mid = l + r >> 1;
        int best = -1;
        push_down(o);
        if (x <= mid) 
            best = find_next(o << 1, l, mid, x, y, lim);
        if (best == -1 && y > mid) 
            best = find_next(o << 1 | 1, mid + 1, r, x, y, lim);

        return best;
    }

    int find_next(int p, int lim) 
    {
        int nxt = find_next(1, 1, V, p, V, lim);
        if (nxt != -1)
            return nxt;
        return find_next(1, 1, V, 1, p, lim);
    }

    inline void add(int p, int v) 
    {
        add(1, 1, V, p, p, v);
    }

    inline void add_tag_density(int pos, int value)
    {
        int L = std::min(V, window_len);
        int pre_pos = std::max(1, pos - L + 1);
        add(1,1,V,pre_pos,pos,value);
        if(pos < L)
        {
            int rest_num = L - pos;
            add(1,1,V,V - rest_num + 1,V,value);
        }
    }
};

struct Segment_tree_add {
    int seg[MAX_DISK_SIZE << 2];
    void set_one(int o, int l, int r) 
    {
        if (l == r) return seg[o] = 1, void();
        int mid = l + r >> 1;
        set_one(o << 1, l, mid), set_one(o << 1 | 1, mid + 1, r);
        seg[o] = seg[o << 1] + seg[o << 1 | 1];
    }

    inline void push_up(int o) 
    {
        seg[o] = seg[o << 1] + seg[o << 1 | 1];
    }

    int add_unit(int o, int l, int r, int p) 
    {
        if (seg[o] < p) return -1;
        if (l == r) {
            seg[o] = 0;
            return l;
        }

        int mid = l + r >> 1;
        int res = -1;
        res = add_unit(o << 1, l, mid, p);
        if (res == -1) {
            res = add_unit(o << 1 | 1, mid + 1, r, p - seg[o << 1]);
        }

        push_up(o);
        return res;
    }

    void add(int o, int l, int r, int p, int v)
    {
        if (l == r) {
            seg[o] += v;
            return ;
        }

        int mid = l + r >> 1;
        if (p <= mid)
            add(o << 1, l, mid, p, v);
        else add(o << 1 | 1, mid + 1, r, p, v);
        push_up(o);
    }

    void modify(int o, int l, int r, int p, int v) 
    {
        if (l == r) {
            seg[o] = v;
            return ;
        }

        int mid = l + r >> 1;
        if (p <= mid)
            modify(o << 1, l, mid, p, v);
        else modify(o << 1 | 1, mid + 1, r, p, v);
        push_up(o);
    }

    void delete_unit(int o, int l, int r, int p)
    {
        if (l == r) return seg[o] = 1, void();
        int mid = l + r >> 1;
        if (p <= mid) 
            delete_unit(o << 1, l, mid, p);
        else delete_unit(o << 1 | 1, mid + 1, r, p);
        push_up(o);
        return;
    }

    int find_next(int o, int l, int r, int x, int y)
    {
        if (x > y) return -1;
        if (seg[o] == 0) return -1;
        if (l == r) return l;

        int mid = l + r >> 1;
        int res = -1;
        if (x <= mid)
            res = find_next(o << 1, l, mid, x, y);
        if (y > mid && res == -1)
            res = find_next(o << 1 | 1, mid + 1, r, x, y);
        return res;
    }

    int find_pre(int o, int l, int r, int x, int y) 
    {
        if (x > y) return -1;
        if (seg[o] == 0) return -1;
        if (l == r) return l;

        int mid = l + r >> 1;
        int res = -1;
        if (y > mid)
            res = find_pre(o << 1 | 1, mid + 1, r, x, y);
        if (x <= mid && res == -1)
            res = find_pre(o << 1, l, mid, x, y);
        return res;
    }

    int find_kth(int o, int l, int r, int x, int y, int k) 
    {
        if (seg[o] < k) return -1;
        if (l == r) return l;

        int mid = l + r >> 1;
        int res = -1;

        if (x <= mid)
            res = find_kth(o << 1, l, mid, x, y, k);
        if (y > mid && res == -1)
            res = find_kth(o << 1 | 1, mid + 1, r, x, y, k - seg[o << 1]);
        return res;
    }

    int query(int o = 1, int l = 1, int r = V, int x = 1, int y = V)
    {
        if (x <= l && y >= r) return seg[o];
        int mid = l + r >> 1, res = 0;
        if (x <= mid)
            res = query(o << 1, l, mid, x, y);
        if(y > mid)
            res += query(o << 1 | 1, mid + 1, r, x, y);
        return res;
    }

    inline void add(int p, int v) 
    {
        add(1, 1, V, p, v);
    }

    inline void modify(int p, int v)
    {
        modify(1, 1, V, p, v);
    }

    inline int query_rest_unit() 
    {
        return seg[1];
    }

    inline int find_next(int x) 
    {
        int res = find_next(1, 1, V, x, V);
        if (res == -1)
            res = find_next(1, 1, V, 1, x);

        return res;
    }

    inline int find_pre(int x) 
    {
        int res = find_pre(1, 1, V, 1, x);
        if (res == -1)
            res = find_pre(1, 1, V, x, V);
        return res;
    }

    inline int find_kth(int k) 
    {
        return find_kth(1, 1, V, 1, V, k);
    }
};
struct DensityManager {
    std::vector<int> request_num;
    std::vector<int> prefix_sum;
    int window_len;
    
    //传入TEST_DENSITY_LEN并初始化
    void init(int len)
    {
        window_len =  std::min(len, V);
        request_num.resize(MAX_DISK_SIZE + 1);
        prefix_sum.resize(MAX_DISK_SIZE + 1);
    }
    
    void add(int pos, int value)
    {
        request_num[pos] += value;
        assert(request_num[pos] >= 0);
    }

    void modify(int pos, int value)
    {
        request_num[pos] = value;
        assert(request_num[pos] >= 0);
    }

    int get(int pos)
    {
        assert(pos > 0 && pos <= V);
        assert(request_num[pos] >= 0);
        return request_num[pos];
    }

    int get_prefix_sum(int pos)
    {
        // std::cerr << "get_prefix_sum begin" << std::endl;

        int cur_prefix_sum = 0,pre_pos = std::max(1,pos - window_len + 1);
        cur_prefix_sum = prefix_sum[pos] - prefix_sum[pre_pos - 1];

        // std::cerr << "cur_prefix_sum: " << cur_prefix_sum << std::endl;

        if(pos < window_len)
        {
            int rest_num = window_len - pos;
            cur_prefix_sum += prefix_sum[V] - prefix_sum[V - rest_num];
        }

        // std::cerr << "get_prefix_sum end" << std::endl;

        return cur_prefix_sum;
    }
    
    int find_max_point(bool find_mid = false)
    {
        int max_point = 1;

        // std::cerr << "find_max_point begin" << std::endl;

        for(int i = 1; i <= V; i++)
            prefix_sum[i] = prefix_sum[i - 1] + request_num[i];

        // std::cerr << "partial_sum end" << std::endl;

        for (int i = 1; i <= V; i++) {
            int window_sum = get_prefix_sum(i);

            // std::cerr << "window_sum: " << window_sum << std::endl;

            if (window_sum > get_prefix_sum(max_point)) {
                max_point = i;
            }
        }

        // std::cerr << "find_max_point end" << std::endl;

        if (!find_mid)
            return get_pre_kth(max_point, window_len);
        else return get_pre_kth(max_point, window_len / 2);
    }
};
struct DISK {
    int pointer; //这个磁盘指针的位置
    int last_read_cnt = 0; //上一次操作往前连续读取的次数
    int last_read_cost = -1; //-1: 上一次不为读取操作 否则为上一次读取操作的花费
    int rest_token;
    int tag_order[MAX_TAG_NUM]; //每个标签在这个磁盘的固定顺序
    int tag_distribution_pointer[MAX_TAG_NUM];
    int tag_distribution_size[MAX_TAG_NUM];
    std::pair <int, int> unit_object[MAX_DISK_SIZE];
    bool inner_tag_inverse[MAX_TAG_NUM];
    bool is_reverse;
    int distribution_strategy;
    int test_density_len = TEST_DENSITY_LEN;
    int tag_num;

    Segment_tree_add all_request;
    Segment_tree_add empty_pos; //维护空位置
    // Segment_tree_max request_num; //维护每个点的request数量
    //Segment_tree_max max_density; //用于获取每个段的request总和
    Segment_tree_max tag_density[MAX_TAG_NUM];
    DensityManager max_density;
    // DensityManager tag_in_disk[MAX_TAG_NUM];
};

DISK disk[MAX_DISK_NUM];
std::mt19937 RAND(666666);

inline int random(int l, int r)
{
    return RAND() % (r - l + 1) + l;
}

struct Predict {
    int add_object;
    int delete_object;
    int read_object;
};

Predict Info[MAX_STAGE][MAX_TAG_NUM];
int max_cur_tag_size[MAX_STAGE][MAX_TAG_NUM], all_time_max_tag_size[MAX_TAG_NUM];
int all_tag_request[MAX_TAG_NUM], test_tag_request[MAX_TAG_NUM];
int go_disk_dist;

/*存储每个对象的unit没有解决的request*/
std::queue<int> unsolve_request[MAX_OBJECT_NUM][MAX_OBJECT_SIZE];
int request_rest_unit[MAX_REQUEST_NUM];
int request_rest_unit_state[MAX_REQUEST_NUM];
std::vector <int> solved_request;

char read_oper[MAX_TOKEN]; //存储读操作
//dp_without_skip存储dp值
//info_without_skip最后一位存储0 pass 1 read,除去最后一位存储上一位的状态
int dp_without_skip[MAX_DISK_SIZE][READ_CNT_STATES], info_without_skip[MAX_DISK_SIZE][READ_CNT_STATES];
int read_cost[READ_CNT_STATES] = {64, 52, 42, 34, 28, 23, 19, 16};
inline void to_next_pos(int& x) 
{
    x = x % V + 1;
}

inline void to_pre_pos(int& x) 
{
    if (x == 1) x = V;
    else --x;
}

int get_pre_pos(int x) {
    if (x == 1) return V;
    return x - 1;
}

/*从x到y的距离*/
inline int get_dist(int x, int y) 
{
    return x <= y? y - x: V - x + y;
}

inline int get_min_dist(int x, int y)
{
    if (x > y) std::swap(x, y);
    return std::min(y - x, V - y + x);
}

inline void distribute_tag_in_disk_front(int disk_id, int stage) 
{
    int rest_unit = disk[disk_id].empty_pos.query_rest_unit() + 10;
    int all_need = 0;
    std::cerr << "DISTRIBUTION : ";
    for (int i = 1; i <= disk[disk_id].tag_num; ++i) {
        int cur_tag = disk[disk_id].tag_order[i];
        all_need += std::max(max_cur_tag_size[stage][cur_tag] / N, 3);
        std::cerr << max_cur_tag_size[stage][cur_tag] << " ";
    }

    std::cerr << std::endl;
    std::cerr << "ALLNEED : " << all_need << " " << rest_unit << std::endl;

    for (int i = 1, pre_distribution = 0; i <= disk[disk_id].tag_num; ++i) {
        int cur_tag = disk[disk_id].tag_order[i];
        // int cur_tag_distribution = std::max(max_cur_tag_size[stage][cur_tag] / N, 3) + 5;
        int cur_tag_distribution = (1.0 * std::max(max_cur_tag_size[stage][cur_tag] / N, 3) / all_need) * rest_unit;
        
        if (pre_distribution + cur_tag_distribution > rest_unit) {
            pre_distribution += cur_tag_distribution;
            //这边如果满了，指针和前面一样
            disk[disk_id].tag_distribution_pointer[cur_tag] = disk[disk_id].empty_pos.find_next(disk[disk_id].tag_distribution_pointer[disk[disk_id].tag_order[i - 1]]);
            continue;
        }
        
        if (!disk[disk_id].inner_tag_inverse[cur_tag]) {
            disk[disk_id].tag_distribution_pointer[cur_tag] = disk[disk_id].empty_pos.find_kth(pre_distribution + 1);
        } else {
            disk[disk_id].tag_distribution_pointer[cur_tag] = disk[disk_id].empty_pos.find_kth(pre_distribution + cur_tag_distribution);
        }

        pre_distribution += cur_tag_distribution;
    }
}

inline void distribute_tag_in_disk_mid(int disk_id, int stage) 
{
    int rest_unit = disk[disk_id].empty_pos.query_rest_unit() + 10;
    int all_need = 0;
    for (int i = 1; i <= disk[disk_id].tag_num; ++i) {
        int cur_tag = disk[disk_id].tag_order[i];
        all_need += std::max(max_cur_tag_size[stage][cur_tag] / N, 3);
        std::cerr << max_cur_tag_size[stage][cur_tag] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "ALLNEED : " << all_need << " " << rest_unit << std::endl;

    for (int i = 1, pre_distribution = 0; i <= disk[disk_id].tag_num; ++i) {
        int cur_tag = disk[disk_id].tag_order[i];
        int cur_tag_distribution = (1.0 * std::max(max_cur_tag_size[stage][cur_tag] / N, 3) / all_need) * rest_unit;
        
        if (pre_distribution + cur_tag_distribution > rest_unit) {
            pre_distribution += cur_tag_distribution;
            //这边如果满了，指针和前面一样
            disk[disk_id].tag_distribution_pointer[cur_tag] = disk[disk_id].tag_distribution_pointer[disk[disk_id].tag_order[i - 1]];
            continue;
        }

        disk[disk_id].tag_distribution_pointer[cur_tag] = pre_distribution + cur_tag_distribution / 2;
        pre_distribution += cur_tag_distribution;
    }
}

inline void distribute_tag_in_disk_by_density(int disk_id, int stage)
{
    auto& cur_disk = disk[disk_id];
    std::vector <DensityManager> tag_density(M + 1);
    for (int i = 1; i <= M; ++i) {
        tag_density[i].init(std::max(10, int(max_cur_tag_size[stage][i] / N / TAG_DENSITY_DIVIDE)));
    }

    for (int i = 1; i <= V; ++i) {
        auto [object_id, _] = cur_disk.unit_object[i];
        int object_tag = objects[object_id].tag;

        if (object_id) {
            tag_density[object_tag].modify(i, cur_disk.max_density.get(i) / UNIT_REQUEST_DIVIDE + 1);
        }
    }

    for (int i = 1; i <= M; ++i) {
        cur_disk.tag_distribution_pointer[i] = tag_density[i].find_max_point();
    }
}

inline void distribute_tag_in_disk_new_version_1(int stage)
{
    std::vector <int> disk_rest_size(N + 1, V);
    std::vector <int> disk_pos(N);
    std::iota(disk_pos.begin(), disk_pos.end(), 1);
    std::vector <Sub_object> all_object;
    int piece_size = V * DIVIDE_TAG_IN_DISK_VERSION1;

    for (int i = 1; i <= M; ++i) {
        int piece_num = std::max(MIN_TAG_NUM_IN_DISK, all_time_max_tag_size[i] / piece_size);
        int cur_size = max_cur_tag_size[stage][i];
        int std_size = (max_cur_tag_size[stage][i] + piece_num - 1) / piece_num;

        for (int j = 1; j <= piece_num; ++j) {
            int cur_p_size = std::min(cur_size, std_size);
            all_object.push_back(Sub_object(i, cur_p_size));
            cur_size -= cur_p_size;
        }

        assert(cur_size == 0);
    }

    std::sort(all_object.begin(), all_object.end(), [&](const Sub_object& x, const Sub_object&  y) {
        return x.size > y.size;
    });

    for (auto [tag, size] : all_object) {
        std::sort(disk_pos.begin(), disk_pos.end(), [&](const int a, const int b) {
            return disk_rest_size[a] > disk_rest_size[b];
        });

        for (int disk_id : disk_pos) {
            // assert(disk_rest_size[disk_id] >= size);
            if (disk[disk_id].tag_distribution_size[tag] == 0) {
                disk[disk_id].tag_distribution_size[tag] = size;
                disk[disk_id].tag_order[++disk[disk_id].tag_num] = tag;
                disk_rest_size[disk_id] -= size;
                // std::cerr << "DISTRIBUTE : " << disk_id << " " << tag << std::endl;
                break;
            }
        }
    }

    for (int i = 1; i <= N; ++i) {
        auto& cur_disk = disk[i];
        int tag_num = cur_disk.tag_num;
        
        std::shuffle(cur_disk.tag_order + 1, cur_disk.tag_order + 1 + tag_num, RAND);
        int all_need = V - disk_rest_size[i];

        for (int j = 1, pre_distribution = 0; j <= tag_num; ++j) {
            int cur_tag = cur_disk.tag_order[j];
            int rest_unit = V;

            cur_disk.inner_tag_inverse[cur_disk.tag_order[j]] = RAND() & 1;
            cur_disk.tag_density[cur_tag].preference_left = cur_disk.inner_tag_inverse[cur_tag];
            cur_disk.tag_density[cur_tag].init(V);
            cur_disk.tag_density[cur_tag].build();
            int cur_tag_distribution = (1.0 * cur_disk.tag_distribution_size[cur_tag] / all_need) * rest_unit;

            if (pre_distribution + cur_tag_distribution > rest_unit) {
                pre_distribution += cur_tag_distribution;
                cur_disk.tag_distribution_pointer[cur_tag] = cur_disk.tag_distribution_pointer[cur_disk.tag_order[j - 1]];
                continue;
            }

            if (!cur_disk.inner_tag_inverse[cur_tag]) {
                cur_disk.tag_distribution_pointer[cur_tag] = pre_distribution + 1;
            } else {
                cur_disk.tag_distribution_pointer[cur_tag] = pre_distribution + cur_tag_distribution;
            }

            pre_distribution += cur_tag_distribution;
        }
    }
}

/*预处理操作*/
void init() 
{
    int cur_time_out_of_queue = EXTRA_TIME;
    int delta_time_out_of_queue = EXTRA_TIME / NUM_PIECE_QUEUE;
    for(int i = NUM_PIECE_QUEUE; i >= 1; i--)
    {
        time_out_of_queue[i] = cur_time_out_of_queue;
        cur_time_out_of_queue -= delta_time_out_of_queue;
    }


    DISK_MIN_PASS = std::min(DISK_MIN_PASS, V - 1);
    for (int i = 1; i <= all_stage; ++i) {
        for (int j = 1; j <= M; ++j) {
            max_cur_tag_size[i][j] = max_cur_tag_size[i - 1][j] - Info[i - 1][j].delete_object + Info[i][j].add_object;
            all_tag_request[j] += Info[i][j].read_object;
            all_time_max_tag_size[j] = std::max(all_time_max_tag_size[j], max_cur_tag_size[i][j]);
        }
    }
    
    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            tag_size_in_disk[i][j] = 0;
        }
    }
    
    if (USE_NEW_DISTRIBUTION) 
        distribute_tag_in_disk_new_version_1(std::min(PRE_DISTRIBUTION_TIME, all_stage));
    else {
        for (int i = 1; i <= N; ++i) {
            disk[i].tag_num = M;
        }
    }

    // std::cerr << "HERE : " << std::endl;
    for (int i = 1; i <= N; i++) {
        disk[i].pointer = 1;
        disk[i].empty_pos.set_one(1, 1, V);
        // disk[i].request_num.build(1, 1, V);
        
        disk[i].max_density.init(TEST_DENSITY_LEN);

        if (!USE_NEW_DISTRIBUTION) {
            std::iota(disk[i].tag_order + 1, disk[i].tag_order + 1 + M, 1);
            std::shuffle(disk[i].tag_order + 1, disk[i].tag_order + 1 + M, RAND);
            for (int j = 1; j <= M; ++j) {
                test_tag_request[j] = max_cur_tag_size[all_stage][j] + ((RAND() & 1) ? 1 : -1) * random(600, 9000);
                // test_tag_request[j] = all_tag_request[j] + ((RAND() & 1) ? 1 : -1) * random(500, 3000);
            }

            // std::sort(disk[i].tag_order + 1, disk[i].tag_order + 1 + M, [&](const int a, const int b) {
                // return test_tag_request[a] > test_tag_request[b];
            // });
            
            // disk[i].is_reverse = i & 1;
            for (int j = 1; j <= M; ++j) {
                disk[i].inner_tag_inverse[j] = RAND() & 1;
                // disk[i].inner_tag_inverse[j] = disk[i].is_reverse;
            }

            int stage = std::min(PRE_DISTRIBUTION_TIME, all_stage);
            disk[i].distribution_strategy = 1;        
            if (disk[i].distribution_strategy == 1)
                distribute_tag_in_disk_front(i, stage);
            else distribute_tag_in_disk_mid(i, stage);
        }
    }

    if (USE_NEW_DISTRIBUTION) {
        for (int i = 1; i <= N; ++i) {
            int all_ = 0;
            std::cerr << "DISK " << i << " ::   ";
            for (int j = 1; j <= disk[i].tag_num; ++j) {
                std::cerr << disk[i].tag_distribution_size[disk[i].tag_order[j]] << " ";
                all_ += disk[i].tag_distribution_size[disk[i].tag_order[j]];
            }

            std::cerr << "    all : " << all_ << "   rest : " << V << std::endl;
            // std::cerr << std::endl;
        }
    }
    
    // std::cerr << "OK" << std::endl; 
    /*
    
    read_cost[0] = read_cost[1] = 64;
    for (int i = 2; i < 9; ++i) {
        read_cost[i] = read
    }
    for (int i = 8; i >= )
    */
}

inline int get_now_stage(int now_time) 
{
    return std::min((now_time + FRE_PER_SLICING - 1) / FRE_PER_SLICING, all_stage);
}

inline void reset_disk_window_len(int disk_id)
{   
    auto& cur_disk = disk[disk_id];
    int n = cur_disk.tag_num;
    // std::cerr << "PRE : \n";
    for (int i = 1; i <= n; ++i) {
        int j = cur_disk.tag_order[i];
        assert(cur_disk.tag_distribution_size[j] > 0);
        // std::cerr << cur_disk.tag_density[j].find_max_point() << " " << cur_disk.tag_density[j].find_max_value() << std::endl;

        // cur_disk.tag_density[j].reset();
        cur_disk.tag_density[j].init(V);
        cur_disk.tag_density[j].build();
        cur_disk.tag_density[j].window_len = std::max(MIN_TEST_TAG_DENSITY_LEN, int(tag_size_in_disk[j][disk_id] / WRITE_TAG_DENSITY_DIVIDE));
        // std::cerr << cur_disk.tag_density[j].window_len << " ";
    }

    // std::cerr << std::endl;
    // std::vector <int> real(MAX_TAG_NUM + 1, 0);

    // std::cerr << "V : ";
    for (int i = 1; i <= V; ++i) {
        auto [object_id, unit_id] = cur_disk.unit_object[i];
        int tag = objects[object_id].tag;
        if (object_id) {
            // if (i <= 20)
                // std::cerr << tag << " ";
            cur_disk.tag_density[tag].add_tag_density(i, 1);
            // ++real[tag];
        }
    }

    // std::cerr << "END : \n"; 
    for (int i = 1; i <= n; ++i) {
        int j = cur_disk.tag_order[i];
        assert(cur_disk.tag_distribution_size[j] > 0);
        // std::cerr << cur_disk.tag_density[j].find_max_point() << " " << cur_disk.tag_density[j].find_max_value() << std::endl;
    }

    // std::cerr << std::endl;
    // std::cerr << "REAL : \n";
    // for (int i = 1; i <= n; ++i) {
        // int j = cur_disk.tag_order[i];
        // std::cerr << real[j] << " ";
    // }
    // std::cerr << std::endl;
}

void timestamp_action()
{
    int timestamp;
    scanf("%*s%d", &timestamp);
    printf("TIMESTAMP %d\n", timestamp);

    TEST_DENSITY_LEN = std::max(cur_request / CUR_REQUEST_DIVIDE, MIN_TEST_DENSITY_LEN);
    //READ_ROUND_TIME = std::max(TEST_DENSITY_LEN / LEN_TIME_DIVIDE, MIN_ROUND_TIME);
    //READ_ROUND_TIME = 3;

    if (get_now_stage(timestamp) != get_now_stage(timestamp - 1)) {
        std::cerr << "CER_REQUEST : " << cur_request << std::endl;
        std::cerr << "DIST : " << go_disk_dist << std::endl;
        if (!USE_NEW_DISTRIBUTION) {
            for (int i = 1; i <= N; ++i) {
                if (get_now_stage(timestamp) <= PRE_DISTRIBUTION_TIME && get_now_stage(timestamp) % 10 == 0);
                    // distribute_tag_in_disk_front(i, get_now_stage(timestamp));
                if ( get_now_stage(timestamp) > PRE_DISTRIBUTION_TIME && get_now_stage(timestamp) % 3 == 0)
                    distribute_tag_in_disk_by_density(i, get_now_stage(timestamp));
                // if (disk[i].distribution_strategy == 1)
                //     distribute_tag_in_disk_front(i, get_now_stage(timestamp));
                // else 
                //     distribute_tag_in_disk_mid(i, get_now_stage(timestamp));
            }
        } else {

            if (get_now_stage(timestamp) % 2 == 0 && get_now_stage(timestamp) > PRE_DISTRIBUTION_TIME) {
               for (int i = 1; i <= N; ++i) {
                    reset_disk_window_len(i);
                } 
            }
            
        }
    }

    fflush(stdout);
} 

/*未完成的请求*/
std::vector <int> abort_request;

//维护density
// inline void modify_max_density(int disk_id, int pos, int delta_request) 
// {
    // int pre_pos = std::max(1, pos - disk[disk_id].test_density_len + 1);
    // disk[disk_id].max_density.add(1, 1, V, pre_pos, pos, delta_request);
    
    // if (pre_pos != pos - disk[disk_id].test_density_len + 1) {
    //     int rest_num = disk[disk_id].test_density_len - pos;
    //     disk[disk_id].max_density.add(1, 1, V, V - rest_num + 1, V, delta_request);
    // }
// }


inline void modify_unit_request(int disk_id, int pos, int value) 
{
    // int delta_request = disk[disk_id].request_num.modify(1, 1, V, pos, value);
    // if(delta_request != 0) modify_max_density(disk_id, pos, delta_request);
    // disk[disk_id].all_request.modify(pos, value);
    disk[disk_id].max_density.modify(pos, value);
}

inline void add_unit_request(int disk_id, int pos, int ad_num) 
{
    // disk[disk_id].request_num.add(1, 1, V, pos, pos, ad_num);
    // modify_max_density(disk_id, pos, ad_num);
    // disk[disk_id].all_request.add(pos, ad_num);
    disk[disk_id].max_density.add(pos, ad_num);
}

inline void do_object_delete(int object_id) 
{
    //delete pos
    int cur_tag = objects[object_id].tag;
    for (int i = 1; i <= REP_NUM; ++i) {
        for (int j = 1; j <= objects[object_id].size; ++j) {
            auto [disk_id, pos] = objects[object_id].unit_pos[i][j];

            //维护空位置
            disk[disk_id].empty_pos.delete_unit(1, 1, V, pos);

            if (USE_NEW_DISTRIBUTION) {
                disk[disk_id].tag_density[cur_tag].add_tag_density(pos, -1);
                // add_tag_density(disk_id, cur_tag, pos, -1);
            }
            
            //清除request
            modify_unit_request(disk_id, pos, 0);
            disk[disk_id].unit_object[pos] = {0, 0};

            tag_size_in_disk[objects[object_id].tag][disk_id] -= 1;
            //更新tag_set
        }
    }

    //find abort req
    for (int i = 1; i <= objects[object_id].size; ++i) {
        while (!unsolve_request[object_id][i].empty()) {
            int now_request = unsolve_request[object_id][i].front();

            if (request_rest_unit[now_request] > 0) {
                abort_request.push_back(now_request);
                request_rest_unit[now_request] = -1;
            }

            unsolve_request[object_id][i].pop();
        }
    }
}

void delete_action()
{
    int n_delete;
    int abort_num = 0;
    static int _id[MAX_OBJECT_NUM];

    scanf("%d", &n_delete);
    // std::cerr << "statDelete : " << n_delete << std::endl;
    for (int i = 1; i <= n_delete; i++) {
        scanf("%d", &_id[i]);
        // std::cerr << "delete : " << _id[i] << std::endl;
        do_object_delete(_id[i]);
    }
    
    // std::cerr << "END" << std::endl;
    printf("%ld\n", abort_request.size());
    cur_request -= abort_request.size();
    // std::cerr << "NOWNOW :: " << abort_request.size() << std::endl;
    for (int req_id : abort_request) {
        printf("%d\n", req_id);
    }

    abort_request.clear();
    fflush(stdout);
}

inline void write_unit(int object_id, int disk_id, int unit_id, int write_pos, int repeat_id) 
{
    // disk[disk_id].empty_pos.add_unit(1, 1, V, 1);
    disk[disk_id].empty_pos.add(1, 1, V, write_pos, -1);

    if (USE_NEW_DISTRIBUTION) {
        if (disk[disk_id].tag_distribution_size[objects[object_id].tag] > 0)
            disk[disk_id].tag_density[objects[object_id].tag].add_tag_density(write_pos, 1);
            // add_tag_density(disk_id, objects[object_id].tag, write_pos, 1);
        // disk[disk_id].tag_in_disk[].add(write_pos, 1);
    }
    
    disk[disk_id].unit_object[write_pos] = {object_id, unit_id};
    objects[object_id].unit_pos[repeat_id][unit_id] = {disk_id, write_pos};
}

inline int write_unit_in_disk_strategy_1(int disk_id, int tag)
{
    int res = 0;
    if (!disk[disk_id].inner_tag_inverse[tag]) {
        res = disk[disk_id].empty_pos.find_next(disk[disk_id].tag_distribution_pointer[tag]);
        // to_next_pos(disk[disk_id].tag_distribution_pointer[tag]);
    } else {
        res = disk[disk_id].empty_pos.find_pre(disk[disk_id].tag_distribution_pointer[tag]);
        // to_pre_pos(disk[disk_id].tag_distribution_pointer[tag]);
    }

    // std::cerr << "NOW_POS : " << res << std::endl;  

    return res;
}

inline int write_unit_in_disk_strategy_2(int disk_id, int tag)
{
    int pre_pos = disk[disk_id].empty_pos.find_next(disk[disk_id].tag_distribution_pointer[tag]);
    int nxt_pos = disk[disk_id].empty_pos.find_pre(disk[disk_id].tag_distribution_pointer[tag]);
    int pos = nxt_pos;
    if (get_min_dist(pre_pos, disk[disk_id].tag_distribution_pointer[tag]) <= get_min_dist(nxt_pos, disk[disk_id].tag_distribution_pointer[tag]))
        pos = pre_pos;
    return pos;
}

inline int write_unit_in_disk_by_density(int disk_id, int tag)
{
    if (!disk[disk_id].tag_distribution_size[tag]) {
        return disk[disk_id].empty_pos.find_next(1);
    }

    int best_pos = disk[disk_id].tag_density[tag].find_max_point();
    best_pos = get_nxt_kth(best_pos, WRITE_TEST_DENSITY_LEN / 2);
    int pre_pos = disk[disk_id].empty_pos.find_next(best_pos);
    int nxt_pos = disk[disk_id].empty_pos.find_pre(best_pos);
    if (get_dist(pre_pos, best_pos) <= get_dist(best_pos, nxt_pos))
        best_pos = pre_pos;
    else best_pos = nxt_pos;
    return best_pos;
}

inline int write_unit_in_disk_by_density_version2(int disk_id, int tag)
{
    if (!disk[disk_id].tag_distribution_size[tag]) {
        return disk[disk_id].empty_pos.find_next(1);
    }

    int best_pos = disk[disk_id].tag_density[tag].find_max_point();
    if (disk[disk_id].inner_tag_inverse[tag]) {
        best_pos = get_nxt_kth(best_pos, WRITE_TEST_DENSITY_LEN);
        best_pos = disk[disk_id].empty_pos.find_pre(best_pos);
    } else {
        best_pos = disk[disk_id].empty_pos.find_next(best_pos);
    }

    return best_pos;
}

void write_action()
{
    int n_write;
    scanf("%d", &n_write);
    // std::cerr << "n_write: " << n_write << std::endl;
    for (int i = 1; i <= n_write; ++i) {
        // std::cerr << "write : "  << i << std::endl;
        int id, size, tag;
        scanf("%d %d %d", &id, &size, &tag);
        objects[id].size = size;
        objects[id].tag = tag;

        std::vector <int> pos(N);
        std::iota(pos.begin(), pos.end(), 1);
        // std::random_shuffle(pos.begin(), pos.end());
        std::sort(pos.begin(), pos.end(),[&](int x,int y){
            return tag_size_in_disk[tag][x] < tag_size_in_disk[tag][y];
        });

        int now = 0;
        printf("%d\n", id);

        if (USE_NEW_DISTRIBUTION) {
            std::vector <int> disk_request(MAX_DISK_NUM + 1, 0);
            for (int j = 1; j <= N; ++j) {
                disk_request[j] = disk[j].all_request.query();
            }

            // std::sort(pos.begin(), pos.end(),[&](int x,int y){
                // return disk_request[x] < disk_request[y];
            // });
            // std::random_shuffle(pos.begin() + 1, pos.end());
            
            for (int j = 1; j <= REP_NUM; ++j) {
                int disk_id = pos[now];
                while (disk[disk_id].tag_distribution_size[tag] == 0 || disk[disk_id].empty_pos.query_rest_unit() < size) {
                    disk_id = pos[++now];
                    
                    if (now >= N) {
                        std::cerr << "tag : " << tag << std::endl;
                        for (int k = 0; k < N; ++k) {
                            disk_id = pos[k];
                            std::cerr << "DISK : " << disk_id << " " << disk[disk_id].tag_distribution_size[tag] << " " << disk[disk_id].empty_pos.query_rest_unit() << std::endl;
                        }
                        
                        for (int k = 0; k < N; ++k) {
                            disk_id = pos[k];
                            if (disk[disk_id].empty_pos.query_rest_unit() >= size) {
                                now = k;
                                break;
                            }
                        }

                        assert(disk[pos[now]].empty_pos.query_rest_unit() >= size);
                        disk_id = pos[now];

                        std::cerr << "FIND pos ->  : " << disk_id << " " << disk[disk_id].empty_pos.query_rest_unit() << std::endl;
                        break;
                    }
                    
                    assert(now < N);
                }

                printf("%d ", disk_id);
                // std::cerr << "disk_id : " << disk_id << " ";

                tag_size_in_disk[tag][disk_id] += size;
                for (int k = 1; k <= size; ++k) {
                    int nxt = 0;

                    if (now_stage <= PRE_DISTRIBUTION_TIME) {
                        nxt = write_unit_in_disk_strategy_1(disk_id, tag);
                    } else {
                        nxt = write_unit_in_disk_by_density_version2(disk_id, tag);
                        // nxt = write_unit_in_disk_by_density(disk_id, tag);
                    }

                    // std::cerr << "Nxt : " << nxt << " ";
                    write_unit(id, disk_id, k, nxt, j);
                    printf("%d ", nxt);
                }

                printf("\n");
                // std::cerr << std::endl;
                ++now;
            }
        } else {
            // std::cerr << "object_id : " << id << std::endl;
            for (int j = 1; j <= REP_NUM; ++j) {
                int disk_id = pos[now];
                while (disk[disk_id].empty_pos.query_rest_unit() < size) {
                    disk_id = pos[++now];
                    assert(now < N);
                }
                
                printf("%d ", disk_id);
                // std::cerr << "disk_id : " << disk_id << " ";

                tag_size_in_disk[tag][disk_id] += size;

                for (int k = 1; k <= size; ++k) {
                    // int nxt = disk[disk_id].empty_pos.find_next(1, 1, V, 1, V);
                    int nxt = 0;
                    // if (disk[disk_id].distribution_strategy == 1)
                    if (now_stage <= PRE_DISTRIBUTION_TIME)
                        nxt = write_unit_in_disk_strategy_1(disk_id, tag);
                    else  
                        nxt = write_unit_in_disk_strategy_2(disk_id, tag);
                    // std::cerr << nxt << " ";
                    // assert(disk[disk_id].empty_pos.find_next(1, 1, V, 1, V) == nxt + 1);
                    write_unit(id, disk_id, k, nxt, j);
                    printf("%d ", nxt);
                    
                }

                // std::cerr << std::endl;
                printf("\n");
                now += 1;
            }
        }

        
    }
    fflush(stdout);
}

//use
inline void read_unit(int object_id, int unit_id, int time) 
{
    while (!unsolve_request[object_id][unit_id].empty()) {
        int request_id = unsolve_request[object_id][unit_id].front();
        if (request_rest_unit[request_id] > 0)
        {
            --request_rest_unit[request_id];
            request_rest_unit_state[request_id] |= 1 << unit_id;
/*
            for(int j = 1; j <= objects[object_id].size; ++j)
            {
                if((1 << j) & request_rest_unit_state[request_id])
                    continue;
                for(int i = 1; i <= REP_NUM; ++i)
                {
                    int delta_time = time - requests[request_id].request_time;
                    int index = request_queue_id[request_id];
                    auto [disk_id, unit_pos] = objects[object_id].unit_pos[i][j];
                    add_unit_request(disk_id, unit_pos, NUM_PIECE_QUEUE - index + 1);
                }
            }
*/
            if (!request_rest_unit[request_id]) {
                solved_request.push_back(request_id);
            }
        }
        unsolve_request[object_id][unit_id].pop();
    }
}

//use
inline void update_unsolved_request(int request_id, int object_id) 
{
    request_rest_unit[request_id] = objects[object_id].size;
    // std::cerr << "update_unsolved_request : " << request_id << " " << request_rest_unit[request_id] << std::endl;

    for (int j = 1; j <= objects[object_id].size; ++j) {
        unsolve_request[object_id][j].push(request_id);       

        for (int i = 1; i <= REP_NUM; ++i) {
            auto [disk_id, unit_id] = objects[object_id].unit_pos[i][j];
            add_unit_request(disk_id, unit_id, NUM_PIECE_QUEUE);
        }
    }
}

//指针进行一次jump
void do_pointer_jump(DISK &cur_disk, int destination) 
{
    printf("j %d\n", destination);
    cur_disk.pointer = destination;
    cur_disk.rest_token = 0;
    cur_disk.last_read_cnt = 0;
    cur_disk.last_read_cost = -1;
}

/*
指针进行一次pass
1 success
0 fail
*/
int do_pointer_pass(DISK &cur_disk)
{
    if (!cur_disk.rest_token)
        return 0;
    printf("p");
    assert(cur_disk.max_density.get(cur_disk.pointer) == 0);
    ++go_disk_dist;
    cur_disk.pointer = cur_disk.pointer % V + 1;
    cur_disk.rest_token--;
    cur_disk.last_read_cnt = 0;
    cur_disk.last_read_cost = -1;
    return 1;
}
/*
指针进行一次read
1 success
0 fail
*/

inline void read_unit_update(int object_id, int unit_id,int time) 
{
    read_unit(object_id, unit_id, time);
    for (int i = 1; i <= REP_NUM; ++i) {
        auto [disk_id, unit_pos] = objects[object_id].unit_pos[i][unit_id];
        modify_unit_request(disk_id, unit_pos, 0);
    }
}

int do_pointer_read(DISK &cur_disk, int time) 
{
    int read_cost = cur_disk.last_read_cnt?
        std::max(16, (cur_disk.last_read_cost * 4 + 5 - 1) / 5) : 64;
    if (cur_disk.rest_token < read_cost)
        return 0;
    printf("r");
    
    //清除request
    ++go_disk_dist;
    int& pos = cur_disk.pointer;
    auto [object_id, unit_id] = cur_disk.unit_object[pos];
    
    if (object_id != 0) {
        read_unit_update(object_id, unit_id, time);
    }

    pos = pos % V + 1;
    ++cur_disk.last_read_cnt;
    cur_disk.rest_token -= read_cost;
    cur_disk.last_read_cost = read_cost;
    return 1;
}

/*
1 选择pass
0 选择read
*/
bool chosse_pass(DISK &cur_disk, int destination)
{
    int dist = get_dist(cur_disk.pointer, destination);
    if (dist < DISK_MIN_PASS)
        return 0;
    return 1;
}

struct Pointer{
    int pointer;
    Pointer(int p) : pointer(p) {}
    int get_to_nxt()
    {
        return pointer % V + 1;
    }
    void to_nxt()
    {
        pointer = pointer % V + 1;
    }
};

//算出skip不jump的DP值
//warning: 需要保证磁盘大小>=每个时间片token数
std::pair<int, int> DP_read_without_skip_and_jump(DISK &cur_disk, int pointer_pos, int rest_token) {
    //memset(dp_without_skip[pointer_pos], -1, sizeof(dp_without_skip[pointer_pos]));
    for (int j = 0; j < READ_CNT_STATES; ++j)
        dp_without_skip[pointer_pos][j] = -1;
    //assert(dp_without_skip[pointer_pos][j] == -1);
    dp_without_skip[pointer_pos][std::min(cur_disk.last_read_cnt, READ_CNT_STATES - 1)] = rest_token;
    Pointer pointer(pointer_pos);
    int sum_request = 0;
    for (int t = 0; t < rest_token; ++t) {
        int nxt = pointer.get_to_nxt();
        int cur = pointer.pointer;
        memset(dp_without_skip[nxt], -1, sizeof(dp_without_skip[nxt]));
        //choose read
        for (int j = 1; j < READ_CNT_STATES; ++j) {
            dp_without_skip[nxt][j] = dp_without_skip[cur][j - 1] - read_cost[j - 1];
            info_without_skip[nxt][j] = (j - 1) << 1 | 1;
        }
        {
            const int t = READ_CNT_STATES - 1;
            if (dp_without_skip[nxt][t] < dp_without_skip[cur][t] - read_cost[t]) {
                dp_without_skip[nxt][t] = dp_without_skip[cur][t] - read_cost[t];
                info_without_skip[nxt][t] = t << 1 | 1;
            }
        }
        int cur_request = cur_disk.max_density.get(cur);
        //choose pass
        if (cur_request == 0) {
            for (int j = 0; j < READ_CNT_STATES; ++j) {
                //dp_without_skip[nxt][0] = std::min(dp_without_skip[nxt][0], dp_without_skip[cur][j] - 1);
                if (dp_without_skip[nxt][0] < dp_without_skip[cur][j] - 1) {
                    dp_without_skip[nxt][0] = dp_without_skip[cur][j] - 1;
                    info_without_skip[nxt][0] = j << 1;
                }
            }
        }
        bool can_get = 0; //是否能到达nxt
        for (int j = 0; j < READ_CNT_STATES; ++j) {
            if (dp_without_skip[nxt][j] >= 0) {
                can_get = 1;
                break;
            }
        }
        if (nxt == cur_disk.pointer) //不能重复读磁盘多次
            can_get = 0;
        if (can_get) {
            sum_request += cur_request;
            pointer.to_nxt();
        }
        else
            break;
    }
    return std::make_pair(sum_request, pointer.pointer);
}

/*从begin_pos到end_pos需要花费的最小的token*/
std::pair<int, int> DP_read_without_skip_and_jump_range(DISK &cur_disk, int begin_pos, int end_pos) {
    //std::cerr << "DP result: " << "from pos: " << begin_pos << " to " << "pos: " << end_pos << "  ";
    //memset(dp_without_skip[pointer_pos], -1, sizeof(dp_without_skip[pointer_pos]));
    for (int j = 0; j < READ_CNT_STATES; ++j)
        dp_without_skip[begin_pos][j] = INF_TOKEN;
    //assert(dp_without_skip[pointer_pos][j] == -1);
    dp_without_skip[begin_pos][std::min(cur_disk.last_read_cnt, READ_CNT_STATES - 1)] = 0;
    Pointer pointer(begin_pos);
    int sum_request = 0;
    while (pointer.pointer != end_pos) {
        int nxt = pointer.get_to_nxt();
        int cur = pointer.pointer;
        for (int j = 0; j < READ_CNT_STATES; ++j)
            dp_without_skip[nxt][j] = INF_TOKEN;
        //choose read
        for (int j = 1; j < READ_CNT_STATES; ++j) {
            if (dp_without_skip[nxt][j] > dp_without_skip[cur][j - 1] + read_cost[j - 1]) {
                dp_without_skip[nxt][j] = dp_without_skip[cur][j - 1] + read_cost[j - 1];
                info_without_skip[nxt][j] = (j - 1) << 1 | 1;
            }
        }
        {
            const int t = READ_CNT_STATES - 1;
            if (dp_without_skip[nxt][t] > dp_without_skip[cur][t] + read_cost[t]) {
                dp_without_skip[nxt][t] = dp_without_skip[cur][t] + read_cost[t];
                info_without_skip[nxt][t] = t << 1 | 1;
            }
        }
        int cur_request = cur_disk.max_density.get(cur);
        //choose pass
        if (cur_request == 0) {
            for (int j = 0; j < READ_CNT_STATES; ++j) {
                //dp_without_skip[nxt][0] = std::min(dp_without_skip[nxt][0], dp_without_skip[cur][j] - 1);
                if (dp_without_skip[nxt][0] > dp_without_skip[cur][j] + 1) {
                    dp_without_skip[nxt][0] = dp_without_skip[cur][j] + 1;
                    info_without_skip[nxt][0] = j << 1;
                }
            }
        }
        pointer.to_nxt();
    }
    int min_cost_token = INF_TOKEN;
    for (int j = 0; j < 7; ++j)
        min_cost_token = std::min(min_cost_token, dp_without_skip[end_pos][j]);
    //std::cerr << "with cost " << min_cost_token << std::endl;
    return std::make_pair(sum_request, min_cost_token);
}

/*按照dp数组从begin_pos读到end_pos*/
void trace_dp(DISK &cur_disk, int begin_pos, int end_pos, int time) {
    int cur_state = 0;
    for (int j = 0; j < 7; ++j) {
        if (dp_without_skip[end_pos][j] < dp_without_skip[end_pos][cur_state])
            cur_state = j;
    }
    int dp_cost_token = dp_without_skip[end_pos][cur_state];
    std::vector<char> read_sequence;
    while (end_pos != begin_pos) {
        if (info_without_skip[end_pos][cur_state] & 1)
            read_sequence.push_back('r');
        else
            read_sequence.push_back('p');
        cur_state = info_without_skip[end_pos][cur_state] >> 1;
        to_pre_pos(end_pos);
    }
    int check_rest_token_start = cur_disk.rest_token;
    int check_requests = 0;
    reverse(read_sequence.begin(), read_sequence.end());
    for (auto oper : read_sequence) {
        if (oper == 'r') {
            check_requests += cur_disk.max_density.get(cur_disk.pointer);
            do_pointer_read(cur_disk, time);
        }
        else
            do_pointer_pass(cur_disk);
    }
    int check_rest_token_end = cur_disk.rest_token;
    assert (check_rest_token_start - check_rest_token_end == dp_cost_token);
}

void read_without_jump(DISK &cur_disk,int time);
void read_without_jump_dp_and_bf_version(DISK &cur_disk, int time) {
    //int begin_pointer = cur_disk.pointer;
    //prel, prer用来处理
    //std::cerr << "start read_without_jump_dp_and_bf_version" << std::endl;
    int prel = cur_disk.pointer, prer = cur_disk.pointer; //两个指针的距离
    int pre_requests = 0; //[prel~prer)的request总和 不包含prer
    bool can_get_pre = 1;
    //std::cerr << "start choose DP or brute force" << std::endl;
    do {
        while (get_dist(prel, prer) < DISK_MIN_PASS_DP) {
            pre_requests += cur_disk.max_density.get(prer);
            to_next_pos(prer);
        }
        bool has_wide_blank = 0;
        while (pre_requests > 0 && get_dist(cur_disk.pointer, prel) <= cur_disk.rest_token) {
            pre_requests -= cur_disk.max_density.get(prel);
            pre_requests += cur_disk.max_density.get(prer);
            to_next_pos(prel);
            to_next_pos(prer);
        }
        if (pre_requests > 0) {
            break;
        }
        assert(prer != cur_disk.pointer);
        //std::cerr << "start DP" << std::endl;
        int min_cost_token = DP_read_without_skip_and_jump_range(cur_disk, cur_disk.pointer, prel).second;
        //std::cerr << "end DP with cost: " << min_cost_token << std::endl;
        if (min_cost_token + MIN_TOKEN_STOP_DP > cur_disk.rest_token) {
            can_get_pre = 0;
        }
        else {
            trace_dp(cur_disk, cur_disk.pointer, prel, time);
            assert(prel == cur_disk.pointer);
            int check_pass_cnt = 0;
            while (cur_disk.max_density.get(cur_disk.pointer) == 0) {
                if (do_pointer_pass(cur_disk))
                    ++check_pass_cnt;
                else
                    break;    
            }
            /*
            //std::cerr << check_pass_cnt << std::endl;
            //std::cerr << cur_disk.pointer << " " << " " << prel << " " << prer << " " << std::endl;
            //std::cerr << "min_cost_token: " << min_cost_token << " " << cur_disk.max_density.get(cur_disk.pointer) << std::endl;
            //std::cerr << "rest_token: " << " " << cur_disk.rest_token << std::endl;
            //std::cerr << "pre_request: " << " " << pre_requests << std::endl;*/
            assert(check_pass_cnt >= DISK_MIN_PASS_DP || cur_disk.rest_token == 0);
            prel = prer = cur_disk.pointer;
            pre_requests = 0;
        }
    }
    while (can_get_pre && cur_disk.rest_token > 0);
    //std::cerr << "end choose DP or brute force" << std::endl;
    read_without_jump(cur_disk, time);
    //while (do_pointer_read(cur_disk, time));
    //std::cerr << "end read_without_jump_dp_and_bf_version" << std::endl;
    //Pointer cur_pointer(cur_disk.pointer);
    //printf("#\n");
}

void read_without_jump_dp_version(DISK &cur_disk, int time)
{
    int begin_pointer = cur_disk.pointer;
    //std::cerr << "start DP_read_without_skip_and_jump" << std::endl;
    auto [sum_requests, end_pointer] = DP_read_without_skip_and_jump(cur_disk, cur_disk.pointer, 2 * cur_disk.rest_token);
    //std::cerr << "end DP_read_without_skip_and_jump" << std::endl;
    //std::cerr << "end_pointer: " << end_pointer << std::endl;
    //std::cerr << "dist: " << get_dist(begin_pointer, end_pointer) << std::endl;
    int cur_state = 0;
    for (int j = 1; j < READ_CNT_STATES; ++j) {
        if (dp_without_skip[end_pointer][j] > dp_without_skip[end_pointer][cur_state])
            cur_state = j;
    }
    assert(dp_without_skip[end_pointer][cur_state] != -1);
    std::vector<char> read_sequence;
    while (end_pointer != begin_pointer) {
        if (info_without_skip[end_pointer][cur_state] & 1)
            read_sequence.push_back('r');
        else
            read_sequence.push_back('p');
        cur_state = info_without_skip[end_pointer][cur_state] >> 1;
        to_pre_pos(end_pointer);
    }
    int check_requests = 0;
    reverse(read_sequence.begin(), read_sequence.end());
    //go_disk_dist += read_sequence.size();
    for (auto oper : read_sequence) {
        if (oper == 'r') {
            check_requests += cur_disk.max_density.get(cur_disk.pointer);
            if (!do_pointer_read(cur_disk, time))
                break;
        }
        else
            do_pointer_pass(cur_disk);
    }
    //std::cerr << check_requests<< " " << sum_requests << std::endl;
    //assert(check_requests == sum_requests);
    printf("#\n");
}

void read_without_jump(DISK &cur_disk,int time)
{
    // 每次都重新计算窗口总和，暴力维护，避免出现sum_of_request < cur_request_num的情况
    auto calc_window_sum = [&](int start_pos) {
        int sum = cur_disk.max_density.get(start_pos);
        Pointer temp(start_pos);
        int count = 1;
        while(count < DISK_MIN_PASS) {
            temp.to_nxt();
            sum += cur_disk.max_density.get(temp.pointer);
            count++;
        }
        return std::make_pair(sum, temp.pointer);
    };
    
    // 初始窗口计算
    auto [sum_of_request, fast_pointer_pos] = calc_window_sum(cur_disk.pointer);
    Pointer fast_pointer(fast_pointer_pos);
    int cur_request_num = cur_disk.max_density.get(cur_disk.pointer);
    
    bool have_rest_token = true;

    while (have_rest_token) {
        // std::cerr << "pre_pre : " << cur_disk.pointer << std::endl;
        if (sum_of_request == 0 || cur_disk.last_read_cnt == 0) {
            // std::cerr << "IN : " << cur_disk.pointer << std::endl;
            assert(cur_request_num >= 0);
            assert(sum_of_request >= 0);
            assert(cur_request_num <= sum_of_request);

            while(cur_request_num == 0 && have_rest_token)
            {
                if(!do_pointer_pass(cur_disk))
                {
                    have_rest_token = false;
                    break;
                }

                // 重新计算窗口
                auto [new_sum, new_fast_pos] = calc_window_sum(cur_disk.pointer);
                sum_of_request = new_sum;
                fast_pointer = Pointer(new_fast_pos);
                cur_request_num = cur_disk.max_density.get(cur_disk.pointer);
            }
        }
        assert(cur_request_num >= 0);
        assert(sum_of_request >= 0);
        assert(cur_request_num <= sum_of_request);

        while(sum_of_request > 0 && have_rest_token)
        {
            if(!do_pointer_read(cur_disk, time))
            {
                have_rest_token = false;
                break;
            }

            // 重新计算窗口
            auto [new_sum, new_fast_pos] = calc_window_sum(cur_disk.pointer);
            sum_of_request = new_sum;
            fast_pointer = Pointer(new_fast_pos);
            cur_request_num = cur_disk.max_density.get(cur_disk.pointer);
        }
        
    }

/*
    while(cur_disk.rest_token > 0)
    {
        // std::cerr << "cur_disk.rest_token: " << cur_disk.rest_token << std::endl;
        // std::cerr << "cur_disk.point: " << cur_disk.pointer << std::endl;
        // std::cerr << "nxt_p: " << nxt_p << std::endl;
        // std::cerr << "choose_pass? " << chosse_pass(cur_disk, nxt_p) << std::endl;
    
        int nxt_p = cur_disk.request_num.find_next(cur_disk.pointer, 1);
        if (nxt_p == -1)
                    break;
        if (chosse_pass(cur_disk, nxt_p)) {
            while (cur_disk.pointer != nxt_p)
                if (!do_pointer_pass(cur_disk))
                    break;
        }
        else {
            while (cur_disk.pointer != nxt_p)
                if (!do_pointer_read(cur_disk))
                    break;
        }
        if (!do_pointer_read(cur_disk))
            break;
    }
*/
    // std::cerr << "end: " << sum_of_request << std::endl;

    printf("#\n");
    //int p = disk[cur_disk].request_num.find_next()
}

void read_action(int time)
{
    for (int cur_disk_id = 1; cur_disk_id <= N; ++cur_disk_id)
        disk[cur_disk_id].rest_token = G;

    int n_read;
    int request_id, object_id;
    scanf("%d", &n_read);
    // std::cerr << "n_read : " << n_read << std::endl;
    cur_request += n_read;
    for (int i = 1; i <= n_read; i++) {
        scanf("%d%d", &request_id, &object_id);
        // std::cerr << "request_id : " << request_id << " " << "ob : " << object_id << std::endl;
        requests[request_id].object_id = object_id;
        requests[request_id].request_time = time;
        requests[request_id].request_id = request_id;
        request_queue_in_time_order[1].push(requests[request_id]);
        request_queue_id[request_id] = 1;
        update_unsolved_request(request_id, object_id);
    }

    // std::cerr << "in read_action: finish read" << std::endl;
    // std::cerr << "in read_action: start move pointer" << std::endl;

    //磁头移动操作
    const int DIST_NOT_JUMP = G;
    for (int cur_disk_id = 1; cur_disk_id <= N; ++cur_disk_id) {
        // std::cerr << "cur_disk_id: " << cur_disk_id << std::endl;
        DISK &cur_disk = disk[cur_disk_id];
        if (time % random(READ_ROUND_TIME, READ_ROUND_TIME) == 1) {
            int p = cur_disk.max_density.find_max_point();
            int ans_p = p == -1? -1: DP_read_without_skip_and_jump(cur_disk, p, TEST_READ_TIME * cur_disk.rest_token).first;
            int ans_now = DP_read_without_skip_and_jump(cur_disk, cur_disk.pointer, TEST_READ_TIME * cur_disk.rest_token).first;
            /*
            if (cur_disk.max_density.get(p) * JUMP_VISCOSITY <= cur_disk.max_density.get(cur_disk.pointer))
                p = cur_disk.pointer;
            */
                // std::cerr << "max_point: " << p << std::endl;

                if (p == -1 || get_dist(cur_disk.pointer, p) <= G * JUMP_VISCOSITY || ans_p < ans_now * 1.6) { //如果距离足够近
                
                // std::cerr << "start read_without_jump" << std::endl;
                
                if (USE_DP) {
                    if (USE_DP == DP_VERSION1)
                        read_without_jump_dp_version(cur_disk, time);
                    else {
                        if (USE_DP == DP_VERSION2)
                            read_without_jump_dp_and_bf_version(cur_disk, time);
                        else
                            assert(0);
                    }
                }
                else
                    read_without_jump(cur_disk, time);
            }
            else 
                do_pointer_jump(cur_disk, p);
        } else {
            if (USE_DP) {
                if (USE_DP == DP_VERSION1)
                    read_without_jump_dp_version(cur_disk, time);
                else {
                    if (USE_DP == DP_VERSION2)
                        read_without_jump_dp_and_bf_version(cur_disk, time);
                    else
                        assert(0);
                }
            }
            else
                read_without_jump(cur_disk, time);
        }
           
    }
    // std::cerr << "in read_action: end move pointer" << std::endl;

    //solved request
    printf("%ld\n", solved_request.size());
    cur_request -= solved_request.size();
    // std::cerr << "SOLSOLSOLS : " << solved_request.size() << std::endl;
    for (int request_id : solved_request) {
        // std::cerr << request_id << " ";
        printf("%d\n", request_id);
    }
    // std::cerr << std::endl;

    solved_request.clear();
    fflush(stdout);
}

inline void update_request_num(int time) {
    for(int i = 1; i <= NUM_PIECE_QUEUE; i++)
    {
        while (!request_queue_in_time_order[i].empty() && request_queue_in_time_order[i].front().request_time < time - time_out_of_queue[i]) {
            _Request now_request = request_queue_in_time_order[i].front();
            request_queue_in_time_order[i].pop();
            if(request_rest_unit[now_request.request_id] <= 0) continue;
            if(i < NUM_PIECE_QUEUE)
            {
                request_queue_in_time_order[i + 1].push(now_request);
                request_queue_id[now_request.request_id] = i + 1;
            }
            for (int j = 1; j <= objects[now_request.object_id].size; ++j) {   
                if(((1 << j) & request_rest_unit_state[now_request.request_id]))
                        continue;
                for (int i = 1; i <= REP_NUM; ++i) {
                    auto [disk_id, unit_id] = objects[now_request.object_id].unit_pos[i][j];
                    // add_unit_request(disk_id, unit_id, -(objects[now_request.object_id].size - request_rest_unit[now_request.request_id] + 1));
                    add_unit_request(disk_id, unit_id, -1);
                }
            }
        }    
    }
    
}

int main()
{
    // std::cerr << "start input global information" << std::endl;
    scanf("%d%d%d%d%d", &T, &M, &N, &V, &G);
    // srand(666666);
    //srand(time(0) ^ clock());

    all_stage = (T - 1) / FRE_PER_SLICING + 1;
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= all_stage; j++) {
            scanf("%d", &Info[j][i].delete_object); 
            // scanf("%*d");
        }
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= all_stage; j++) {
            scanf("%d", &Info[j][i].add_object);
            // scanf("%*d");
        }
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= all_stage; j++) {
            scanf("%d", &Info[j][i].read_object);
            // scanf("%*d");
        }
    }

    // std::cerr << "end input global information" << std::endl;

    init();
    printf("OK\n");
    fflush(stdout);

    for (int t = 1; t <= T + EXTRA_TIME; t++) {
        now_stage = get_now_stage(t);
        update_request_num(t);

        // std::cerr << "start time " << t << std::endl;
        // std::cerr << "start timestamp_action" <<std::endl;

        timestamp_action();

        // std::cerr << "end timestamp_action" <<std::endl;
        // std::cerr << "start delete_action" <<std::endl;
        delete_action();

        // std::cerr << "end delete_action" <<std::endl;
        // std::cerr << "start write_action" <<std::endl;

        write_action();

        // std::cerr << "end write_action" <<std::endl;
        // std::cerr << "start read_action" <<std::endl;
        read_action(t);

        // std::cerr << "end read_action" <<std::endl;
        // std::cerr << "end time " << t << std::endl;
    }

    return 0;
}   