// ─────────────────────────────────────────
// memory.js · Companion 记忆系统
// ─────────────────────────────────────────
// 数据结构：
// {
//   id:              string   唯一ID
//   content:         string   记忆内容
//   type:            'task' | 'wish' | 'fact'
//   tags:            string[] 标签（由AI提取）
//   status:          'active' | 'done' | 'expired'
//   created_at:      number   时间戳
//   last_triggered:  number   上次被提起的时间戳
//   trigger_count:   number   被触发次数
//   due:             number?  截止时间（仅task）
// }
// ─────────────────────────────────────────

const Memory = (() => {

  const STORAGE_KEY = 'companion_memories';

  // ── 基础读写 ──────────────────────────────

  function _load() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch {
      return [];
    }
  }

  function _save(list) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  }

  // ── ID生成 ────────────────────────────────

  function _id() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  }

  // ── 增 ────────────────────────────────────

  /**
   * 添加一条记忆
   * @param {object} params
   * @param {'task'|'wish'|'fact'} params.type
   * @param {string} params.content
   * @param {string[]} [params.tags]
   * @param {number} [params.due]  截止时间戳（task专用）
   * @returns {object} 新记忆对象
   */
  function add({ type, content, tags = [], due = null }) {
    const list = _load();
    const now = Date.now();
    const item = {
      id:             _id(),
      content,
      type,
      tags,
      status:         'active',
      created_at:     now,
      last_triggered: now,
      trigger_count:  0,
      due,
    };
    list.push(item);
    _save(list);
    return item;
  }

  // ── 查 ────────────────────────────────────

  /** 获取所有记忆（可按type和status过滤） */
  function getAll({ type = null, status = null } = {}) {
    let list = _load();
    if (type)   list = list.filter(m => m.type === type);
    if (status) list = list.filter(m => m.status === status);
    return list;
  }

  /** 按ID查单条 */
  function getById(id) {
    return _load().find(m => m.id === id) || null;
  }

  /** 查active任务 */
  function getTasks()  { return getAll({ type: 'task',  status: 'active' }); }

  /** 查active心愿 */
  function getWishes() { return getAll({ type: 'wish',  status: 'active' }); }

  // ── 改 ────────────────────────────────────

  /**
   * 更新任意字段
   * @param {string} id
   * @param {object} patch
   */
  function update(id, patch) {
    const list = _load();
    const idx  = list.findIndex(m => m.id === id);
    if (idx === -1) return null;
    list[idx] = { ...list[idx], ...patch };
    _save(list);
    return list[idx];
  }

  /** 标记任务完成 */
  function markDone(id) {
    return update(id, { status: 'done' });
  }

  /** 标记过期/归档 */
  function expire(id) {
    return update(id, { status: 'expired' });
  }

  /**
   * 记录一次触发（推荐系统调用）
   * 更新 last_triggered 和 trigger_count
   */
  function recordTrigger(id) {
    const m = getById(id);
    if (!m) return null;
    return update(id, {
      last_triggered: Date.now(),
      trigger_count:  (m.trigger_count || 0) + 1,
    });
  }

  // ── 删 ────────────────────────────────────

  function remove(id) {
    const list = _load().filter(m => m.id !== id);
    _save(list);
  }

  function clearAll() {
    _save([]);
  }

  // ── 推荐引擎 · 时间触发 ───────────────────

  /**
   * 找出"最久没被提起"的记忆
   * 策略：active状态，按 last_triggered 升序排，取前N条
   *
   * @param {object} opts
   * @param {number} [opts.minAgeDays=7]   至少N天未触发才候选
   * @param {number} [opts.limit=3]        返回条数上限
   * @param {'task'|'wish'|null} [opts.type]
   * @returns {object[]}
   */
  function getStalest({ minAgeDays = 7, limit = 3, type = null } = {}) {
    const cutoff = Date.now() - minAgeDays * 86400_000;
    let list = _load().filter(m =>
      m.status === 'active' &&
      m.last_triggered < cutoff &&
      (type ? m.type === type : true)
    );
    // 最久未触发排在前面
    list.sort((a, b) => a.last_triggered - b.last_triggered);
    return list.slice(0, limit);
  }

  /**
   * 获取"今日推荐"：优先返回最久未触发的一条心愿
   * 如果没有心愿则返回最久未触发的任务
   * 如果都没有则返回 null
   *
   * @param {number} [minAgeDays=3]
   * @returns {object|null}
   */
  function getDailyPick(minAgeDays = 3) {
    const wishes = getStalest({ minAgeDays, limit: 1, type: 'wish' });
    if (wishes.length) return wishes[0];
    const tasks  = getStalest({ minAgeDays, limit: 1, type: 'task'  });
    if (tasks.length)  return tasks[0];
    return null;
  }

  // ── 推荐引擎 · 关键词触发 ─────────────────

  /**
   * 用对话文本在记忆库里做简单关键词匹配
   * 返回命中的 active 记忆列表
   *
   * @param {string} text  当前对话内容
   * @param {number} [limit=2]
   * @returns {object[]}
   */
  function matchByKeyword(text, limit = 2) {
    if (!text || !text.trim()) return [];
    const words = text
      .toLowerCase()
      .replace(/[，。！？、\s]+/g, ' ')
      .split(' ')
      .filter(w => w.length >= 2);   // 忽略单字词

    const list = _load().filter(m => m.status === 'active');

    const scored = list.map(m => {
      const target = (m.content + ' ' + m.tags.join(' ')).toLowerCase();
      const hits   = words.filter(w => target.includes(w)).length;
      return { item: m, hits };
    }).filter(x => x.hits > 0);

    scored.sort((a, b) => b.hits - a.hits);
    return scored.slice(0, limit).map(x => x.item);
  }

  // ── 统计 · 周报用 ──────────────────────────

  /**
   * 返回最近N天的统计摘要
   * @param {number} [days=7]
   */
  function getStats(days = 7) {
    const cutoff = Date.now() - days * 86400_000;
    const all    = _load();
    const recent = all.filter(m => m.created_at >= cutoff);

    return {
      total:       all.length,
      active:      all.filter(m => m.status === 'active').length,
      done:        all.filter(m => m.status === 'done').length,
      expired:     all.filter(m => m.status === 'expired').length,
      // 近N天新增
      recentAdded: recent.length,
      recentTasks: recent.filter(m => m.type === 'task').length,
      recentWishes:recent.filter(m => m.type === 'wish').length,
      // 近N天完成
      recentDone:  all.filter(m => m.status === 'done' && m.created_at >= cutoff).length,
    };
  }

  // ── 衰减检查 · 定期调用 ───────────────────

  /**
   * 检查是否有记忆需要"复活确认"
   * 超过 expireDays 未触发的记忆返回出来，让UI询问用户是否还有效
   *
   * @param {number} [expireDays=30]
   * @returns {object[]}
   */
  function getExpiringSoon(expireDays = 30) {
    const cutoff = Date.now() - expireDays * 86400_000;
    return _load().filter(m =>
      m.status === 'active' &&
      m.last_triggered < cutoff
    );
  }

  // ── 调试用 ────────────────────────────────

  /** 在控制台打印当前记忆库（调试用） */
  function debug() {
    const list = _load();
    console.table(list.map(m => ({
      id:      m.id,
      type:    m.type,
      status:  m.status,
      content: m.content.slice(0, 30),
      days_since_trigger: Math.floor((Date.now() - m.last_triggered) / 86400_000),
    })));
    return list;
  }

  // ── 公开接口 ──────────────────────────────

  return {
    add,
    getAll,
    getById,
    getTasks,
    getWishes,
    update,
    markDone,
    expire,
    recordTrigger,
    remove,
    clearAll,
    getStalest,
    getDailyPick,
    matchByKeyword,
    getStats,
    getExpiringSoon,
    debug,
  };

})();
