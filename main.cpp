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
#define MAX_STAGE (50)

const int READ_ROUND_TIME = 10; //一轮读取的时间
const int PRE_DISTRIBUTION_TIME = 15;
const int TEST_DENSITY_LEN = 150;
const int EXTRA_TIME_HALF = 52;
int DISK_MIN_PASS = 6;

struct _Object {
    //(磁盘编号，磁盘内位置)
    std::pair <int, int> unit_pos[REP_NUM + 1][MAX_OBJECT_SIZE];
    int size;
    int tag;
};

struct _Request {
    int object_id;
    int request_time;
    int request_id;
};

int tag_size_in_disk[MAX_TAG_NUM][MAX_DISK_NUM];

_Request requests[MAX_REQUEST_NUM];
std::queue <_Request> request_queue_in_time_order_early,request_queue_in_time_order_late;

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意，objects加了复数
_Object objects[MAX_OBJECT_NUM];

int T, M, N, V, G, all_stage;

struct Segment_tree_max {
    struct Node {
        int max, pos;
        Node() {}
        Node(int _max, int _pos) : max(_max), pos(_pos) {}
        friend Node operator + (const Node& x, const int& num) {
            return Node(x.max + num, x.pos);
        }

        friend bool operator < (const Node& x, const Node& y) {
            return x.max < y.max;
        }
    };

    Node seg[MAX_DISK_SIZE << 2];
    int add_tag[MAX_DISK_SIZE << 2];

    void build(int o, int l, int r) {
        if (l == r) {
            seg[o].pos = l;
            return ;
        }

        int mid = l + r >> 1;
        build(o << 1, l, mid);
        build(o << 1 | 1, mid + 1, r);
        seg[o] = seg[o << 1];
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

    inline void push_up(int o) 
    {
        seg[o] = std::max(seg[o << 1], seg[o << 1 | 1]);
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

    int find_next(int p, int lim) {
        int nxt = find_next(1, 1, V, p, V, lim);
        if (nxt != -1)
            return nxt;
        return find_next(1, 1, V, 1, p, lim);
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

        seg[o] = seg[o << 1] + seg[o << 1 | 1];
        return res;
    }

    void modify(int o, int l, int r, int p, int v) {
        if (l == r) {
            seg[o] += v;
            return ;
        }

        int mid = l + r >> 1;
        if (p <= mid)
            modify(o << 1, l, mid, p, v);
        else modify(o << 1 | 1, mid + 1, r, p, v);
        seg[o] = seg[o << 1] + seg[o << 1 | 1];
    }

    void delete_unit(int o, int l, int r, int p)
    {
        if (l == r) return seg[o] = 1, void();
        int mid = l + r >> 1;
        if (p <= mid) 
            delete_unit(o << 1, l, mid, p);
        else delete_unit(o << 1 | 1, mid + 1, r, p);
        seg[o] = seg[o << 1] + seg[o << 1 | 1];
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
    }

    void modify(int pos, int value)
    {
        request_num[pos] = value;
    }

    int get(int pos)
    {
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
    
    int find_max_point()
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
        
        if(max_point >= window_len) max_point = max_point - window_len + 1;
        else max_point = max_point + V - window_len + 1;

        return max_point;
    }
};
struct DISK {
    Segment_tree_add empty_pos; //维护空位置
    Segment_tree_max request_num; //维护每个点的request数量
    //Segment_tree_max max_density; //用于获取每个段的request总和
    DensityManager max_density;
    int pointer; //这个磁盘指针的位置
    int last_read_cnt = 0; //上一次操作往前连续读取的次数
    int last_read_cost = -1; //-1: 上一次不为读取操作 否则为上一次读取操作的花费
    int rest_token;
    int tag_order[MAX_TAG_NUM]; //每个标签在这个磁盘的固定顺序
    int tag_distribution_pointer[MAX_TAG_NUM];
    std::pair <int, int> unit_object[MAX_DISK_SIZE];
    bool inner_tag_inverse[MAX_TAG_NUM];
    int distribution_strategy;
    int test_density_len = TEST_DENSITY_LEN;
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

Predict Info[MAX_TAG_NUM][MAX_STAGE];
int max_cur_tag_size[MAX_STAGE][MAX_TAG_NUM];
int all_tag_request[MAX_TAG_NUM], test_tag_request[MAX_TAG_NUM];

/*存储每个对象的unit没有解决的request*/
std::queue<int> unsolve_request[MAX_OBJECT_NUM][MAX_OBJECT_SIZE];
int request_rest_unit[MAX_REQUEST_NUM];
int request_rest_unit_state[MAX_REQUEST_NUM];
std::vector <int> solved_request;

inline void get_next_pos(int& x) 
{
    x = x % V + 1;
}

inline void get_pre_pos(int& x) 
{
    if (x == 1) x = V;
    else --x;
}

/*从x到y的距离*/
inline int get_dist(int x, int y) 
{
    return x <= y? y - x: V - x + y;
}

inline void distribute_tag_in_disk_front(int disk_id, int stage) 
{
    int rest_unit = disk[disk_id].empty_pos.query_rest_unit() + 10;
    int all_need = 0;
    for (int i = 1; i <= M; ++i) {
        int cur_tag = disk[disk_id].tag_order[i];
        all_need += std::max(max_cur_tag_size[stage][cur_tag] / N, 3);
    }

    for (int i = 1, pre_distribution = 0; i <= M; ++i) {
        int cur_tag = disk[disk_id].tag_order[i];
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
    int rest_unit = disk[disk_id].empty_pos.query_rest_unit();
    for (int i = 1, pre_distribution = 0; i <= M; ++i) {
        int cur_tag = disk[disk_id].tag_order[i];
        int cur_tag_distribution = std::max(max_cur_tag_size[stage][cur_tag] / N, 3) + 5;
        
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

/*预处理操作*/
void init() 
{
    DISK_MIN_PASS = std::min(DISK_MIN_PASS, V - 1);
    for (int i = 1; i <= all_stage; ++i) {
        for (int j = 1; j <= M; ++j) {
            max_cur_tag_size[i][j] = max_cur_tag_size[i - 1][j] - Info[i - 1][j].delete_object + Info[i][j].add_object;
            all_tag_request[j] += Info[i][j].read_object;
        }
    }
    
    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            tag_size_in_disk[i][j] = 0;
        }
    }

    for (int i = 1; i <= N; i++) {
        disk[i].pointer = 1;
        disk[i].empty_pos.set_one(1, 1, V);

        // disk[i].request_num.build(1, 1, V);
        
        disk[i].max_density.init(TEST_DENSITY_LEN);
        std::iota(disk[i].tag_order + 1, disk[i].tag_order + 1 + M, 1);
        std::shuffle(disk[i].tag_order + 1, disk[i].tag_order + 1 + M, RAND);

        for (int j = 1; j <= M; ++j) {
            test_tag_request[j] = max_cur_tag_size[all_stage][j] + ((RAND() & 1) ? 1 : -1) * random(200, 3000);
            // test_tag_request[j] = all_tag_request[j] + ((RAND() & 1) ? 1 : -1) * random(500, 3000);
        }

        // std::sort(disk[i].tag_order + 1, disk[i].tag_order + 1 + M, [&](const int a, const int b) {
            // return test_tag_request[a] > test_tag_request[b];
        // });

        for (int j = 1; j <= M; ++j) {
            disk[i].inner_tag_inverse[j] = RAND() & 1;
        }

        int stage = std::min(PRE_DISTRIBUTION_TIME, all_stage);
        disk[i].distribution_strategy = 1;

        if (disk[i].distribution_strategy == 1)
            distribute_tag_in_disk_front(i, stage);
        else distribute_tag_in_disk_mid(i, stage);
    }
    
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

void timestamp_action()
{
    int timestamp;
    scanf("%*s%d", &timestamp);
    printf("TIMESTAMP %d\n", timestamp);

    if (get_now_stage(timestamp) > PRE_DISTRIBUTION_TIME && get_now_stage(timestamp) != get_now_stage(timestamp - 1)) {
        for (int i = 1; i <= N; ++i) {
            if (disk[i].distribution_strategy == 1)
                distribute_tag_in_disk_front(i, get_now_stage(timestamp));
            else 
                distribute_tag_in_disk_mid(i, get_now_stage(timestamp));
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
    
    disk[disk_id].max_density.modify(pos, value);
}

inline void add_unit_request(int disk_id, int pos, int ad_num) 
{
    // disk[disk_id].request_num.add(1, 1, V, pos, pos, ad_num);
    // modify_max_density(disk_id, pos, ad_num);
    
    disk[disk_id].max_density.add(pos, ad_num);
}

inline void do_object_delete(int object_id) 
{
    //delete pos
    for (int i = 1; i <= REP_NUM; ++i) {
        for (int j = 1; j <= objects[object_id].size; ++j) {
            auto [disk_id, pos] = objects[object_id].unit_pos[i][j];

            //维护空位置
            disk[disk_id].empty_pos.delete_unit(1, 1, V, pos);
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
    disk[disk_id].empty_pos.modify(1, 1, V, write_pos, -1);
    disk[disk_id].unit_object[write_pos] = {object_id, unit_id};
    objects[object_id].unit_pos[repeat_id][unit_id] = {disk_id, write_pos};
}

inline int write_unit_in_disk_strategy_1(int disk_id, int tag)
{
    int res = 0;
    if (!disk[disk_id].inner_tag_inverse[tag]) {
        res = disk[disk_id].empty_pos.find_next(disk[disk_id].tag_distribution_pointer[tag]);
        // get_next_pos(disk[disk_id].tag_distribution_pointer[tag]);
    } else {
        res = disk[disk_id].empty_pos.find_pre(disk[disk_id].tag_distribution_pointer[tag]);
        // get_pre_pos(disk[disk_id].tag_distribution_pointer[tag]);
    }

    return res;
}

inline int write_unit_in_disk_strategy_2(int disk_id, int tag)
{
    int pos = 0;
    if (RAND() & 1) {
        pos = disk[disk_id].empty_pos.find_next(disk[disk_id].tag_distribution_pointer[tag]);
    } else {
        pos = disk[disk_id].empty_pos.find_pre(disk[disk_id].tag_distribution_pointer[tag]);
    }

    return pos;
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
                int pos = 0;
                if (disk[disk_id].distribution_strategy == 1)
                    pos = write_unit_in_disk_strategy_1(disk_id, tag);
                else  
                    pos = write_unit_in_disk_strategy_2(disk_id, tag);
                // assert(disk[disk_id].empty_pos.find_next(1, 1, V, 1, V) == nxt + 1);
                write_unit(id, disk_id, k, pos, j);
                printf("%d ", pos);
                // std::cerr << nxt << " ";
            }

            // std::cerr << std::endl;
            printf("\n");
            now += 1;
        }
    }
    fflush(stdout);
}

//use
inline void read_unit(int object_id, int unit_id) 
{
    while (!unsolve_request[object_id][unit_id].empty()) {
        int request_id = unsolve_request[object_id][unit_id].front();
        if(request_rest_unit[request_id] > 0)
        {
            --request_rest_unit[request_id];
            request_rest_unit_state[request_id] |= 1 << unit_id;
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
            add_unit_request(disk_id, unit_id, 2);
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

inline void read_unit_update(int object_id, int unit_id) 
{
    read_unit(object_id, unit_id);
    for (int i = 1; i <= REP_NUM; ++i) {
        auto [disk_id, unit_pos] = objects[object_id].unit_pos[i][unit_id];
        modify_unit_request(disk_id, unit_pos, 0);
    }
}

int do_pointer_read(DISK &cur_disk) 
{
    int read_cost = cur_disk.last_read_cnt?
        std::max(16, (cur_disk.last_read_cost * 4 + 5 - 1) / 5) : 64;
    if (cur_disk.rest_token < read_cost)
        return 0;
    printf("r");
    
    //清除request
    int& pos = cur_disk.pointer;
    auto [object_id, unit_id] = cur_disk.unit_object[pos];
    
    if (object_id != 0) {
        read_unit_update(object_id, unit_id);
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

void read_without_jump(DISK &cur_disk)
{
    // while(do_pointer_read(cur_disk))
    // {
    //     // std::cerr << "disk_pointer: " << cur_disk.pointer << std::endl;
    // }
    // printf("#\n");
    // return ;

    Pointer fast_pointer = Pointer(cur_disk.pointer);
    int sum_of_request = 0;
    int cur_request_num = cur_disk.max_density.get(cur_disk.pointer);
    while(get_dist(cur_disk.pointer, fast_pointer.pointer) < DISK_MIN_PASS)
    {
        // std::cerr << "fast_pointer.pointer: " << fast_pointer.pointer << std::endl;

        sum_of_request += cur_disk.max_density.get(fast_pointer.pointer);
        fast_pointer.to_nxt();
    }
    sum_of_request += cur_disk.max_density.get(fast_pointer.pointer);

    bool have_rest_token = true;

    // std::cerr << "start: " << sum_of_request << std::endl;

    while (have_rest_token) {
        if (sum_of_request == 0 || cur_disk.last_read_cnt == 0) {
            assert(cur_request_num >= 0);
            assert(cur_request_num <= sum_of_request);
            while(cur_request_num == 0 && have_rest_token)
            {
                sum_of_request -= cur_request_num;
                if(!do_pointer_pass(cur_disk))
                {
                    have_rest_token = false;
                    break;
                }

                // std::cerr << "disk_pointer: " << cur_disk.rest_token << std::endl;
                // std::cerr << "cur_request_num: " << cur_request_num << std::endl;

                fast_pointer.to_nxt();
                sum_of_request += cur_disk.max_density.get(fast_pointer.pointer);
                cur_request_num = cur_disk.max_density.get(cur_disk.pointer);
            }
        }
            assert(cur_request_num >= 0);
            assert(cur_request_num <= sum_of_request);

            while(sum_of_request > 0 && have_rest_token)
            {
                sum_of_request -= cur_request_num;
                if(!do_pointer_read(cur_disk))
                {
                    have_rest_token = false;
                    break;
                }

            //     // std::cerr << "disk_pointer: " << cur_disk.pointer << std::endl;

                // std::cerr << "disk_rest_token: " << cur_disk.rest_token << std::endl;


                fast_pointer.to_nxt();
                sum_of_request += cur_disk.max_density.get(fast_pointer.pointer);
                cur_request_num = cur_disk.max_density.get(cur_disk.pointer);
            }
    }

    // while(cur_disk.rest_token > 0)
    // {
    //     // std::cerr << "cur_disk.rest_token: " << cur_disk.rest_token << std::endl;
    //     // std::cerr << "cur_disk.point: " << cur_disk.pointer << std::endl;
    //     // std::cerr << "nxt_p: " << nxt_p << std::endl;
    //     // std::cerr << "choose_pass? " << chosse_pass(cur_disk, nxt_p) << std::endl;
    
    //     int nxt_p = cur_disk.request_num.find_next(cur_disk.pointer, 1);
    //     if (nxt_p == -1)
    //                 break;
    //     if (chosse_pass(cur_disk, nxt_p)) {
    //         while (cur_disk.pointer != nxt_p)
    //             if (!do_pointer_pass(cur_disk))
    //                 break;
    //     }
    //     else {
    //         while (cur_disk.pointer != nxt_p)
    //             if (!do_pointer_read(cur_disk))
    //                 break;
    //     }
    //     if (!do_pointer_read(cur_disk))
    //         break;
    // }

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
    for (int i = 1; i <= n_read; i++) {
        scanf("%d%d", &request_id, &object_id);
        // std::cerr << "request_id : " << request_id << " " << "ob : " << object_id << std::endl;
        requests[request_id].object_id = object_id;
        requests[request_id].request_time = time;
        requests[request_id].request_id = request_id;
        request_queue_in_time_order_late.push(requests[request_id]);
        update_unsolved_request(request_id, object_id);
    }

    // std::cerr << "in read_action: finish read" << std::endl;
    // std::cerr << "in read_action: start move pointer" << std::endl;

    //磁头移动操作
    const int DIST_NOT_JUMP = G;
    for (int cur_disk_id = 1; cur_disk_id <= N; ++cur_disk_id) {
        // std::cerr << "cur_disk_id: " << cur_disk_id << std::endl;
        DISK &cur_disk = disk[cur_disk_id];
        if (time % READ_ROUND_TIME == 1) {
            int p = cur_disk.max_density.find_max_point();
            /*
            if (cur_disk.max_density.get(p) * JUMP_VISCOSITY <= cur_disk.max_density.get(cur_disk.pointer))
                p = cur_disk.pointer;
            */
                // std::cerr << "max_point: " << p << std::endl;

            if (p == -1 || get_dist(cur_disk.pointer, p) <= G * 0.9) { //如果距离足够近
                
                // std::cerr << "start read_without_jump" << std::endl;
                read_without_jump(cur_disk);
            }
            else
                do_pointer_jump(cur_disk, p);
        }
        else
            read_without_jump(cur_disk);
    }
    // std::cerr << "in read_action: end move pointer" << std::endl;

    //solved request
    printf("%ld\n", solved_request.size());
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
    while (!request_queue_in_time_order_late.empty() && request_queue_in_time_order_late.front().request_time < time - EXTRA_TIME_HALF) {
        _Request now_request = request_queue_in_time_order_late.front();
        request_queue_in_time_order_late.pop();
        if(request_rest_unit[now_request.request_id] <= 0) continue;
        request_queue_in_time_order_early.push(now_request);
        for (int i = 1; i <= REP_NUM; ++i) {
            
            for (int j = 1; j <= objects[now_request.object_id].size; ++j) {
                if(((1 << j) & request_rest_unit_state[now_request.request_id]))
                    continue;
                auto [disk_id, unit_id] = objects[now_request.object_id].unit_pos[i][j];
                add_unit_request(disk_id, unit_id, -1);
            }
        }
    }
    while (!request_queue_in_time_order_early.empty() && request_queue_in_time_order_early.front().request_time < time - EXTRA_TIME) {
        _Request now_request = request_queue_in_time_order_early.front();
        request_queue_in_time_order_early.pop();
        if(request_rest_unit[now_request.request_id] <= 0) continue;
        for (int i = 1; i <= REP_NUM; ++i) {
            for (int j = 1; j <= objects[now_request.object_id].size; ++j) {
                if(((1 << j) & request_rest_unit_state[now_request.request_id]))
                    continue;
                auto [disk_id, unit_id] = objects[now_request.object_id].unit_pos[i][j];
                add_unit_request(disk_id, unit_id, -1);
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
            scanf("%d", &Info[i][j].delete_object); 
            // scanf("%*d");
        }
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= all_stage; j++) {
            scanf("%d", &Info[i][j].add_object);
            // scanf("%*d");
        }
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= all_stage; j++) {
            scanf("%d", &Info[i][j].read_object);
            // scanf("%*d");
        }
    }

    // std::cerr << "end input global information" << std::endl;

    init();
    printf("OK\n");
    fflush(stdout);

    for (int t = 1; t <= T + EXTRA_TIME; t++) {
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