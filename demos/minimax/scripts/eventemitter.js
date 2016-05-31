// eventemitter.js version 0.1.1.
(function() {
  // An EventEmitter makes it easy to emit and subscribe to events.
  function EventEmitter() {
    this._listeners = {};
  }

  EventEmitter.makeSubclass = function(p) {
    var keys = Object.keys(EventEmitter.prototype);
    for (var i = 0, len = keys.length; i < len; ++i) {
      var key = keys[i];
      p[key] = EventEmitter.prototype[key];
    }
  };

  EventEmitter.prototype.listeners = function(name) {
    return (this._listeners[name] || []).slice();
  };

  EventEmitter.prototype.addListener = function(name, listener) {
    var listeners = this._listeners[name];
    if (!listeners) {
      this._listeners[name] = [listener];
    } else {
      listeners.push(listener);
    }
  };

  EventEmitter.prototype.removeListener = function(name, listener) {
    var listeners = this._listeners[name];
    if (!listeners) {
      return;
    }
    var idx = listeners.indexOf(listener);
    if (idx < 0) {
      return;
    }
    listeners.splice(idx, 1);
  };

  EventEmitter.prototype.on = EventEmitter.prototype.addListener;

  EventEmitter.prototype.once = function(name, listener) {
    var f;
    f = function() {
      listener.apply(this, arguments);
      this.removeListener(name, f);
    }.bind(this);
    this.addListener(name, f);
  };

  EventEmitter.prototype.removeAllListeners = function(name) {
    if ('undefined' === typeof name) {
      this._listeners = {};
    } else {
      this._listeners[name] = [];
    }
  };

  EventEmitter.prototype.emit = function(name) {
    var listeners = this.listeners(name);
    var eventArgs = Array.prototype.slice.call(arguments, 1);
    for (var i = 0, len = listeners.length; i < len; ++i) {
      listeners[i].apply(null, eventArgs);
    }
  };

  window.EventEmitter = EventEmitter;

})();
