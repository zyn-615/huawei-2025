#include <bits/stdc++.h>

#define MAX_DISK_NUM (10 + 1)
#define MAX_DISK_SIZE (16384 + 1)
#define MAX_REQUEST_NUM (30000000 + 1)
#define MAX_OBJECT_NUM (100000 + 1)
#define REP_NUM (3)
#define FRE_PER_SLICING (1800)
#define EXTRA_TIME (105)
#define MAX_OBJECT_SIZE (5 + 1)

typedef struct Request_ {
    int object_id;
    int prev_id;
    bool is_done;
} Request;

typedef struct Object_ {
    int replica[REP_NUM + 1];
    int* unit[REP_NUM + 1];
    int size;
    int last_request_point;
    bool is_delete;
} Object;

Request request[MAX_REQUEST_NUM];
Object object[MAX_OBJECT_NUM];

struct _Object {
    //(磁盘编号，磁盘内位置)
    std::pair <char, short> unit_pos[REP_NUM + 1][MAX_OBJECT_SIZE];
    char size;
    char tag;
    //读入的时候注意赋值给char型
};

struct _Request {
    int object_id;
};

_Request requests[MAX_REQUEST_NUM];

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意，objects加了复数
_Object objects[MAX_OBJECT_NUM];

int T, M, N, V, G;
int disk[MAX_DISK_NUM][MAX_DISK_SIZE];
int disk_point[MAX_DISK_NUM];

/*存储每个对象的unit没有解决的request*/
std::queue<int> unsolve_request[MAX_OBJECT_NUM][MAX_OBJECT_SIZE];
char request_rest_unit[MAX_REQUEST_NUM];
std::vector <int> solved_request;

void timestamp_action()
{
    int timestamp;
    scanf("%*s%d", &timestamp);
    printf("TIMESTAMP %d\n", timestamp);

    fflush(stdout);
}

/*未完成的请求*/
std::vector <int> abort_request;
inline void do_object_delete(int object_id) 
{
    //delete pos

    //find abort req
    for (int i = 1; i <= objects[object_id].size; ++i) {
        while (!unsolve_request[object_id][i].empty()) {
            int now_request = unsolve_request[object_id][i].front();

            if (request_rest_unit[now_request] != -1) {
                abort_request.push_back(now_request);
                request_rest_unit[now_request] = -1;
            }
        }
    }
}

void do_object_delete(const int* object_unit, int* disk_unit, int size)
{
    for (int i = 1; i <= size; i++) {
        disk_unit[object_unit[i]] = 0;
    }
}

void delete_action()
{
    int n_delete;
    int abort_num = 0;
    static int _id[MAX_OBJECT_NUM];

    scanf("%d", &n_delete);
    for (int i = 1; i <= n_delete; i++) {
        scanf("%d", &_id[i]);
    }

    for (int i = 1; i <= n_delete; i++) {
        int id = _id[i];
        int current_id = object[id].last_request_point;
        while (current_id != 0) {
            if (request[current_id].is_done == false) {
                abort_num++;
            }
            current_id = request[current_id].prev_id;
        }
    }

    printf("%d\n", abort_request.size());
    for (int req_id : abort_request) {
        printf("%d\n", req_id);
    }

    // printf("%d\n", abort_num);
    // for (int i = 1; i <= n_delete; i++) {
    //     int id = _id[i];
    //     int current_id = object[id].last_request_point;
    //     while (current_id != 0) {
    //         if (request[current_id].is_done == false) {
    //             printf("%d\n", current_id);
    //         }
    //         current_id = request[current_id].prev_id;
    //     }
    //     for (int j = 1; j <= REP_NUM; j++) {
    //         do_object_delete(object[id].unit[j], disk[object[id].replica[j]], object[id].size);
    //     }
    //     object[id].is_delete = true;
    // }

    fflush(stdout);
}

void do_object_write(int* object_unit, int* disk_unit, int size, int object_id)
{
    int current_write_point = 0;
    for (int i = 1; i <= V; i++) {
        if (disk_unit[i] == 0) {
            disk_unit[i] = object_id;
            object_unit[++current_write_point] = i;
            if (current_write_point == size) {
                break;
            }
        }
    }

    assert(current_write_point == size);
}

/*存储磁盘的每一段*/
struct spare_block {
    int l, r, len;
    inline bool operator < (const spare_block &ano) {
        return len < ano.len;
    }
};

std::multiset<spare_block> remain[MAX_DISK_NUM];

void write_action()
{
    int n_write;
    scanf("%d", &n_write);
    for (int i = 1; i <= n_write; ++i) {

    }
    /*
    int n_write;
    scanf("%d", &n_write);
    for (int i = 1; i <= n_write; i++) {
        int id, size;
        scanf("%d%d%*d", &id, &size);
        object[id].last_request_point = 0;
        for (int j = 1; j <= REP_NUM; j++) {
            object[id].replica[j] = (id + j) % N + 1;
            object[id].unit[j] = static_cast<int*>(malloc(sizeof(int) * (size + 1)));
            object[id].size = size;
            object[id].is_delete = false;
            do_object_write(object[id].unit[j], disk[object[id].replica[j]], size, id);
        }

        printf("%d\n", id);
        for (int j = 1; j <= REP_NUM; j++) {
            printf("%d", object[id].replica[j]);
            for (int k = 1; k <= size; k++) {
                printf(" %d", object[id].unit[j][k]);
            }
            printf("\n");
        }
    }
    */
    fflush(stdout);
}

inline void read_unit(int id, int unit_id) 
{
    while (!unsolve_request[id][unit_id].empty()) {
        int now_request = unsolve_request[id][unit_id].front();
        --request_rest_unit[now_request];
        if (!request_rest_unit[now_request]) {
            solved_request.push_back(now_request);
        }

        unsolve_request[id][unit_id].pop();
    }
}

inline void update_unsolved_request(int request_id, int object_id) 
{
    request_rest_unit[request_id] = objects[object_id].size;
    for (int i = 1; i <= objects[object_id].size; ++i) {
        unsolve_request[object_id][i].push(request_id);
    }
}

void read_action()
{
    int n_read;
    int request_id, object_id;
    scanf("%d", &n_read);
    for (int i = 1; i <= n_read; i++) {
        scanf("%d%d", &request_id, &object_id);
        requests[request_id].object_id = object_id;
        update_unsolved_request(request_id, object_id);

        // request[request_id].object_id = object_id;
        // request[request_id].prev_id = object[object_id].last_request_point;
        // object[object_id].last_request_point = request_id;
        // request[request_id].is_done = false;
    }

    //磁头移动操作


    //solved request
    printf("%d\n", solved_request.size());
    for (int request_id : solved_request) {
        printf("%d\n", request_id);
    }

    solved_request.clear();

    // static int current_request = 0;
    // static int current_phase = 0;
    // if (!current_request && n_read > 0) {
    //     current_request = request_id;
    // }
    // if (!current_request) {
    //     for (int i = 1; i <= N; i++) {
    //         printf("#\n");
    //     }
    //     printf("0\n");
    // } else {
    //     current_phase++;
    //     object_id = request[current_request].object_id;
    //     for (int i = 1; i <= N; i++) {
    //         if (i == object[object_id].replica[1]) {
    //             if (current_phase % 2 == 1) {
    //                 printf("j %d\n", object[object_id].unit[1][current_phase / 2 + 1]);
    //             } else {
    //                 printf("r#\n");
    //             }
    //         } else {
    //             printf("#\n");
    //         }
    //     }

    //     if (current_phase == object[object_id].size * 2) {
    //         if (object[object_id].is_delete) {
    //             printf("0\n");
    //         } else {
    //             printf("1\n%d\n", current_request);
    //             request[current_request].is_done = true;
    //         }
    //         current_request = 0;
    //         current_phase = 0;
    //     } else {
    //         printf("0\n");
    //     }
    // }

    fflush(stdout);
}

void clean()
{
    for (auto& obj : object) {
        for (int i = 1; i <= REP_NUM; i++) {
            if (obj.unit[i] == nullptr)
                continue;
            free(obj.unit[i]);
            obj.unit[i] = nullptr;
        }
    }
}

int main()
{
    scanf("%d%d%d%d%d", &T, &M, &N, &V, &G);

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= (T - 1) / FRE_PER_SLICING + 1; j++) {
            scanf("%*d");
        }
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= (T - 1) / FRE_PER_SLICING + 1; j++) {
            scanf("%*d");
        }
    }

    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= (T - 1) / FRE_PER_SLICING + 1; j++) {
            scanf("%*d");
        }
    }

    printf("OK\n");
    fflush(stdout);

    for (int i = 1; i <= N; i++) {
        disk_point[i] = 1;
    }

    for (int t = 1; t <= T + EXTRA_TIME; t++) {
        timestamp_action();
        delete_action();
        write_action();
        read_action();
    }
    clean();

    return 0;
}